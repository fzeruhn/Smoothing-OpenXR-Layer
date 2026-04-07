#include "pch.h"
#include "runtime_thread.h"
#include "holding_pen.h"
#include "framework/dispatch.gen.h"
#include <log.h>
#include <fmt/format.h>
#include <thread>
#include <chrono>

namespace openxr_api_layer {

using namespace log;

bool RuntimeThread::Start(Config config) {
    if (m_running.load()) return true;
    if (!config.api || config.session == XR_NULL_HANDLE) {
        Log("[RuntimeThread] Invalid config: null api or session\n");
        return false;
    }
    if (!config.holdingPen || !config.holdingPen->IsInitialized()) {
        Log("[RuntimeThread] Invalid config: HoldingPen not initialized\n");
        return false;
    }
    if (!config.queueMutex || config.queue == VK_NULL_HANDLE) {
        Log("[RuntimeThread] Invalid config: null queue or queueMutex\n");
        return false;
    }

    m_config = std::move(config);
    m_stop.store(false);
    m_thread = std::thread([this]() { ThreadMain(); });
    Log("[RuntimeThread] Started\n");
    return true;
}

void RuntimeThread::Stop() {
    m_stop.store(true);
    if (m_thread.joinable()) {
        m_thread.join();
    }
    m_running.store(false);
    Log("[RuntimeThread] Stopped\n");
}

bool RuntimeThread::GetSynthesizedFrameState(XrFrameState& frameState) {
    std::lock_guard<std::mutex> lock(m_frameStateMutex);
    if (!m_frameStateValid) return false;
    frameState.type                   = XR_TYPE_FRAME_STATE;
    frameState.next                   = nullptr;
    frameState.predictedDisplayTime   = m_lastDisplayTime + m_displayPeriod;
    frameState.predictedDisplayPeriod = m_displayPeriod;
    frameState.shouldRender           = XR_TRUE;
    return true;
}

bool RuntimeThread::InitVulkanResources() {
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = m_config.queueFamilyIndex;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(m_config.device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        Log("[RuntimeThread] Failed to create command pool\n");
        return false;
    }

    VkCommandBufferAllocateInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbInfo.commandPool        = m_commandPool;
    cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(m_config.device, &cbInfo, &m_commandBuffer) != VK_SUCCESS) {
        Log("[RuntimeThread] Failed to allocate command buffer\n");
        return false;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(m_config.device, &fenceInfo, nullptr, &m_copyFence) != VK_SUCCESS) {
        Log("[RuntimeThread] Failed to create copy fence\n");
        return false;
    }

    Log("[RuntimeThread] Vulkan resources initialized\n");
    return true;
}

void RuntimeThread::CleanupVulkanResources() {
    if (m_config.device == VK_NULL_HANDLE) return;
    if (m_copyFence)   { vkDestroyFence(m_config.device, m_copyFence, nullptr);         m_copyFence   = VK_NULL_HANDLE; }
    if (m_commandPool) { vkDestroyCommandPool(m_config.device, m_commandPool, nullptr); m_commandPool = VK_NULL_HANDLE; }
    m_commandBuffer = VK_NULL_HANDLE;
}

void RuntimeThread::ThreadMain() {
    m_running.store(true);
    Log("[RuntimeThread] Thread started\n");

    if (!InitVulkanResources()) {
        Log("[RuntimeThread] Vulkan init failed; thread exiting\n");
        m_running.store(false);
        return;
    }

    while (!m_stop.load()) {
        // ── Step 1: pace to compositor ─────────────────────────────────────
        XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState    frameState{XR_TYPE_FRAME_STATE};
        const XrResult waitResult =
            m_config.waitFrame(m_config.session, &waitInfo, &frameState);
        if (XR_FAILED(waitResult)) {
            Log(fmt::format("[RuntimeThread] xrWaitFrame failed: {}\n",
                            static_cast<int>(waitResult)));
            std::this_thread::sleep_for(std::chrono::milliseconds(11));
            continue;
        }

        // Publish timing data to app thread
        {
            std::lock_guard<std::mutex> lock(m_frameStateMutex);
            if (m_lastDisplayTime > 0) {
                const XrDuration measured = frameState.predictedDisplayTime - m_lastDisplayTime;
                if (measured > 0 && measured < 50'000'000LL) {  // 0–50 ms sanity check
                    m_displayPeriod = measured;
                }
            }
            m_lastDisplayTime = frameState.predictedDisplayTime;
            m_frameStateValid = true;
        }

        TraceLoggingWrite(g_traceProvider, "RuntimeThread_WaitFrame",
                          TLArg(frameState.predictedDisplayTime, "DisplayTime"),
                          TLArg(frameState.shouldRender, "ShouldRender"));

        // ── Step 2: begin frame ────────────────────────────────────────────
        XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
        const XrResult beginResult =
            m_config.beginFrame(m_config.session, &beginInfo);
        if (XR_FAILED(beginResult) && beginResult != XR_FRAME_DISCARDED) {
            Log(fmt::format("[RuntimeThread] xrBeginFrame failed: {}\n",
                            static_cast<int>(beginResult)));
        }

        if (!frameState.shouldRender) {
            SubmitEmptyFrame(frameState.predictedDisplayTime);
            continue;
        }

        // ── Step 3: consume a holding-pen slot ────────────────────────────
        m_config.holdingPen->PollSlots();
        const int32_t slot = m_config.holdingPen->GetFreshestReadySlot();

        TraceLoggingWrite(g_traceProvider, "RuntimeThread_Frame",
                          TLArg(frameState.predictedDisplayTime, "DisplayTime"),
                          TLArg(slot, "HoldingSlot"),
                          TLArg(m_config.holdingPen->GetDebugState().c_str(), "PenState"));

        // Phase 3 Step B: always submit empty frame.
        // Step C (Task 6) will call SubmitColorFrame instead when slot >= 0.
        SubmitEmptyFrame(frameState.predictedDisplayTime);
        if (slot >= 0) {
            m_config.holdingPen->ReleaseSlot(slot);
        }
    }

    // Drain GPU before destroying Vulkan objects
    {
        std::lock_guard<std::mutex> lock(*m_config.queueMutex);
        vkQueueWaitIdle(m_config.queue);
    }
    CleanupVulkanResources();
    m_running.store(false);
    Log("[RuntimeThread] Thread exited\n");
}

void RuntimeThread::SubmitEmptyFrame(XrTime displayTime) {
    XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
    endInfo.displayTime          = displayTime;
    endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    endInfo.layerCount           = 0;
    endInfo.layers               = nullptr;
    const XrResult result = m_config.endFrame(m_config.session, &endInfo);

    TraceLoggingWrite(g_traceProvider, "RuntimeThread_EmptyFrame",
                      TLArg(displayTime, "DisplayTime"),
                      TLArg(static_cast<int>(result), "Result"));

    if (XR_FAILED(result)) {
        Log(fmt::format("[RuntimeThread] xrEndFrame (empty) failed: {}\n",
                        static_cast<int>(result)));
    }
}

bool RuntimeThread::SubmitColorFrame(int32_t holdingSlot, XrTime displayTime) {
    // Implemented in Task 6 (Step C).
    (void)holdingSlot;
    (void)displayTime;
    return false;
}

} // namespace openxr_api_layer
