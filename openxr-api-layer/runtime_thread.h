#pragma once

#include "holding_pen.h"

#include <openxr/openxr.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace openxr_api_layer {

class OpenXrApi;

// Owns the OpenXR compositor pacing loop on a dedicated std::thread.
//
// Loop: xrWaitFrame → xrBeginFrame → decision → xrEndFrame
//   Path A (on-time):    consumes the latest HoldingPen slot, blits it to the
//                        injection swapchain, and submits to xrEndFrame.
//   Path B (miss):       pose-warps the last submitted frame (Phase 3: unwarped
//                        cached frame — full CUDA warp deferred to Phase 4).
//   Stall (>10 misses):  submits an empty layer list (black frame) to signal
//                        that the app has hung.
//
// Thread safety:
//   All vkQueueSubmit calls are protected by the shared g_queueMutex defined
//   in layer.cpp, which HoldingPen also uses.
//
// Recursion guard:
//   g_isRuntimeThread is set to true in ThreadBody(). This causes
//   xrAcquireSwapchainImage and xrEndFrame hooks in layer.cpp to fast-path
//   to the downstream dispatch, bypassing frame-capture logic.
class RuntimeThread {
  public:
    // api:                  The layer's OpenXrApi dispatch object (*this from OpenXrLayer).
    // session:              The active XrSession.
    // holdingPen:           Already-constructed HoldingPen (owned by OpenXrLayer).
    // injectionSwapchain:   FrameInjection synthetic swapchain used for presentation.
    // injectionVkImages:    VkImage array backing the injection swapchain (from
    //                       m_injectionVulkanImages in layer.cpp). May be empty
    //                       until the injection swapchain is fully enumerated.
    // localSpace:           XrSpace used to build the projection layer. Pass
    //                       XR_NULL_HANDLE if not yet created; update via SetLocalSpace().
    // device/runtimeQueue:  Vulkan handles for blit submits.
    // waitFn/releaseFn:     Function pointers captured by layer.cpp in xrCreateSession.
    RuntimeThread(OpenXrApi& api,
                  XrSession session,
                  HoldingPen& holdingPen,
                  XrSwapchain injectionSwapchain,
                  const std::vector<VkImage>& injectionVkImages,
                  XrSpace localSpace,
                  VkDevice device,
                  VkQueue runtimeQueue,
                  uint32_t runtimeQueueFamily,
                  uint32_t imageWidth,
                  uint32_t imageHeight,
                  PFN_xrBeginFrame beginFrameFn,
                  PFN_xrWaitSwapchainImage waitFn,
                  PFN_xrReleaseSwapchainImage releaseFn);

    ~RuntimeThread();

    RuntimeThread(const RuntimeThread&) = delete;
    RuntimeThread& operator=(const RuntimeThread&) = delete;

    // Signals the thread to stop and blocks until it exits.
    void RequestShutdownAndJoin();

    // Update the injection image array after the swapchain is enumerated.
    // Safe to call before the first frame is submitted; not safe mid-run.
    void SetInjectionImages(const std::vector<VkImage>& images) {
        m_injectionVkImages = images;
    }

    // Update the local XrSpace after it is created in xrCreateSession.
    void SetLocalSpace(XrSpace space) { m_localSpace = space; }

    // Read by the app-thread synthetic xrWaitFrame path. Thread-safe (atomic).
    int64_t GetLastDisplayTime() const {
        return m_lastDisplayTime.load(std::memory_order_acquire);
    }
    int64_t GetDisplayPeriod() const {
        return m_displayPeriod.load(std::memory_order_acquire);
    }

    // Called by the app thread's synthetic xrWaitFrame to block until the
    // RuntimeThread has called xrBeginFrame for the current cycle.
    // This ensures xrAcquireSwapchainImage is always preceded by a real
    // xrBeginFrame on the compositor side, keeping SteamVR's state valid.
    void WaitForBeginFrame() {
        std::unique_lock<std::mutex> lk(m_beginMutex);
        m_beginCv.wait(lk, [this] {
            return m_beginFrameReady || m_shutdownRequested.load(std::memory_order_relaxed);
        });
        m_beginFrameReady = false;
    }

  private:
    void ThreadBody();

    // Path A: blit the holding pen slot image into the injection swapchain and
    // submit it to xrEndFrame. waitSemaphore is the slot's copyDone binary
    // semaphore; pass VK_NULL_HANDLE when resubmitting a cached slot (Path B).
    // consumedFence is attached to the blit vkQueueSubmit so it fires when the
    // GPU finishes reading the slot, safely freeing it for the app thread.
    void SubmitSlotImage(XrTime displayTime,
                         VkImage sourceImage,
                         VkSemaphore waitSemaphore,
                         XrPosef pose,
                         VkFence consumedFence = VK_NULL_HANDLE,
                         uint64_t sourceFrameId = 0);

    // Path B: submit the last cached frame (Phase 3 stub — no CUDA warp yet).
    void SubmitCachedFrame(XrTime displayTime);

    // Stall path: submit an empty layer list (compositor shows black).
    void SubmitBlackFrame(XrTime displayTime);

    // Strip XrCompositionLayerDepthInfoKHR chains from a projection view array.
    static void StripDepthChains(XrCompositionLayerProjectionView* views, uint32_t count);

    // ---- dependencies ----
    OpenXrApi&           m_api;
    XrSession            m_session;
    HoldingPen&          m_holdingPen;
    XrSwapchain          m_injectionSwapchain;
    std::vector<VkImage> m_injectionVkImages;
    XrSpace              m_localSpace{XR_NULL_HANDLE};

    // ---- Vulkan ----
    VkDevice      m_device{VK_NULL_HANDLE};
    VkQueue       m_runtimeQueue{VK_NULL_HANDLE};
    uint32_t      m_runtimeQueueFamily{0};
    uint32_t      m_imageWidth{0};
    uint32_t      m_imageHeight{0};

    VkCommandPool   m_blitPool{VK_NULL_HANDLE};
    VkCommandBuffer m_blitCmd{VK_NULL_HANDLE};
    // CPU-side fence for blit completion. Used by SubmitSlotImage to ensure the
    // injection image is fully written before xrReleaseSwapchainImage hands it
    // to SteamVR. When consumedFence is provided (Path A) it doubles as this
    // fence; m_blitDoneFence is the fallback for Path B (cached resubmit).
    VkFence         m_blitDoneFence{VK_NULL_HANDLE};

    // ---- OpenXR function pointers ----
    PFN_xrBeginFrame            m_xrBeginFrame{nullptr};
    PFN_xrWaitSwapchainImage    m_xrWaitSwapchainImage{nullptr};
    PFN_xrReleaseSwapchainImage m_xrReleaseSwapchainImage{nullptr};

    // ---- thread state ----
    std::atomic<bool>         m_shutdownRequested{false};
    std::thread               m_thread;
    std::mutex                m_beginMutex;
    std::condition_variable   m_beginCv;
    bool                      m_beginFrameReady{false};

    // Written by RuntimeThread after each real xrWaitFrame; read by the app-thread
    // synthetic xrWaitFrame path in layer.cpp.
    std::atomic<int64_t> m_lastDisplayTime{0};
    std::atomic<int64_t> m_displayPeriod{11111111LL}; // ~90 Hz default

    // ---- per-iteration state (runtime thread only) ----
    std::optional<HoldingPen::ReadySlot> m_lastSubmittedSlot;
    int      m_consecutivePathBCount{0};
    uint32_t m_rtFrameCount{0};
};

} // namespace openxr_api_layer
