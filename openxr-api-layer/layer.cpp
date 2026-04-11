// MIT License
//
// << insert your own copyright here >>
//
// Based on https://github.com/mbucchia/OpenXR-Layer-Template.
// Copyright(c) 2022-2023 Matthieu Bucchianeri
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright noticeand this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "pch.h"
#include "layer.h"
#include "depth_provider.h"
#include "frame_broker.h"
#include "frame_context.h"
#include "frame_injection.h"
#include "holding_pen.h"
#include "runtime_thread.h"
#include "frame_synthesizer.h"
#include "gpu_pipeline_kernels.h"
#include "hole_filler.h"
#include "ofa_pipeline.h"
#include "pose_provider.h"
#include "vulkan_cuda_interop.h"
#include <log.h>
#include <util.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <cuda.h>

#define VK_USE_PLATFORM_WIN32_KHR
#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr_platform.h>
#include <map>       // For tracking image indices
#include <vector>

// TODO (Item 4): Uncomment when pose warp is fully integrated
// #include "pose_warp.h"
// #include "pose_warp_math.h"


namespace openxr_api_layer {

    // Thread-local flag set to true inside the RuntimeThread's std::thread body.
    // When true, all OpenXR hook overrides fast-path to the downstream dispatch,
    // bypassing layer logic to prevent re-entrant frame capture or deadlocks.
    // Pattern mirrors FrameInjection::s_creatingSwapchain.
    thread_local bool g_isRuntimeThread = false;

    // Mutex protecting the shared VkQueue. Most Vulkan apps expose only one queue,
    // so both the app thread (HoldingPen copy submits) and the runtime thread
    // (blit + presentation submits) must serialize their vkQueueSubmit calls.
    std::mutex g_queueMutex;

    using namespace log;

    // Vulkan result check for internal layer use (not to be confused with CHECK_XRCMD).
    // Returns from the enclosing bool function on failure.
#define CHECK_VK_LAYER(call) \
    do { \
        VkResult _vr = (call); \
        if (_vr != VK_SUCCESS) { \
            Log(fmt::format("[ERROR] Vulkan call failed: {} (VkResult={})\n", #call, static_cast<int>(_vr))); \
            return false; \
        } \
    } while (0)

    // Our API layer implement these extensions, and their specified version.
    const std::vector<std::pair<std::string, uint32_t>> advertisedExtensions = {};

    // Initialize these vectors with arrays of extensions to block and implicitly request for the instance.
    const std::vector<std::string> blockedExtensions = {};
    const std::vector<std::string> implicitExtensions = {};

    class OpenXrLayer;

    XrResult XRAPI_CALL xrGetVulkanInstanceExtensionsKHR_intercept(XrInstance instance,
                                                                    XrSystemId systemId,
                                                                    uint32_t bufferCapacityInput,
                                                                    uint32_t* bufferCountOutput,
                                                                    char* buffer);

    XrResult XRAPI_CALL xrGetVulkanDeviceExtensionsKHR_intercept(XrInstance instance,
                                                                  XrSystemId systemId,
                                                                  uint32_t bufferCapacityInput,
                                                                  uint32_t* bufferCountOutput,
                                                                  char* buffer);

    // App-thread xrBeginFrame intercept: when RuntimeThread owns pacing, return
    // XR_SUCCESS immediately (the RuntimeThread calls the real xrBeginFrame).
    XrResult XRAPI_CALL xrBeginFrame_intercept(XrSession session,
                                               const XrFrameBeginInfo* frameBeginInfo);

    class VulkanFrameProcessor {
      public:
        static constexpr uint32_t kCommandBufferRingSize = 3;

        struct StageValidity {
            bool preWarp{false};
            bool ofa{false};
            bool depthLinearization{false};
            bool stereoAdaptation{false};
            bool synthesis{false};
            bool holeFill{false};

            bool FullyValid() const {
                return preWarp && ofa && depthLinearization && stereoAdaptation && synthesis && holeFill;
            }
        };

        VulkanFrameProcessor(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, uint32_t queueFamilyIndex)
            : m_physicalDevice(physicalDevice), m_device(device), m_queue(queue) {
            m_valid = Initialize(queueFamilyIndex);
        }

        bool IsValid() const {
            return m_valid;
        }

        void PollCompletedWork() {
            for (uint32_t slot = 0; slot < kCommandBufferRingSize; ++slot) {
                auto& work = m_slotWork[slot];
                if (!work.submitted) {
                    continue;
                }

                const VkResult fenceStatus = vkGetFenceStatus(m_device, m_submitFences[slot]);
                if (fenceStatus == VK_SUCCESS) {
                    work.submitted = false;
                    if (work.frameId > m_lastCompletedFrameId) {
                        m_lastCompletedFrameId = work.frameId;
                        m_latestStageValidity = work.validity;
                        m_latestOutputColor = work.output;
                    }
                }
            }
        }

        const StageValidity& GetLatestStageValidity() const {
            return m_latestStageValidity;
        }

        bool HasFullyValidOutput() const {
            return m_latestStageValidity.FullyValid() && m_latestOutputColor != VK_NULL_HANDLE;
        }

        VkImage GetLatestOutputImage() const {
            return m_latestOutputColor;
        }

        uint32_t GetFenceBusyCount() const {
            return m_fenceBusyCount;
        }

        bool IsLivePipelineReady() const {
            return m_livePipelineReady;
        }

        const char* GetLastPipelineFailureReason() const {
            return m_lastPipelineFailureReason.c_str();
        }

        bool WaitForCopyCompletion() const {
            if (!m_valid) {
                Log("[ERROR] WaitForCopyCompletion called on invalid VulkanFrameProcessor.\n");
                return false;
            }
            if (m_submitIndex == 0) {
                return true;
            }

            const uint32_t lastSlot = (m_submitIndex - 1) % kCommandBufferRingSize;
            const VkResult waitResult = vkWaitForFences(m_device, 1, &m_submitFences[lastSlot], VK_TRUE, UINT64_MAX);
            if (waitResult != VK_SUCCESS) {
                Log(fmt::format("[ERROR] Vulkan call failed: vkWaitForFences (VkResult={})\n", static_cast<int>(waitResult)));
                return false;
            }
            return true;
        }

      private:
        void ResetLivePipelineResources() {
            if (m_grayCurrent != 0) {
                cuMemFree(m_grayCurrent);
                m_grayCurrent = 0;
            }
            if (m_grayPrevious != 0) {
                cuMemFree(m_grayPrevious);
                m_grayPrevious = 0;
            }
            m_holeFiller.reset();
            m_frameSynthesizer.reset();
            m_ofaPipeline.reset();
            m_cudaToVk.reset();
            m_vkToCuda.reset();
            m_stageOutputColor.reset();
            m_stagePreviousColor.reset();
            m_stageCurrentColor.reset();
            m_livePipelineReady = false;
            m_pendingCudaOutput = false;
            m_stageCurrentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            m_stagePreviousLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            m_stageOutputLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        void ResetCudaContext() {
            if (m_cuStream != nullptr) {
                cuStreamDestroy(m_cuStream);
                m_cuStream = nullptr;
            }
            if (m_cuContext != nullptr) {
                cuCtxSetCurrent(nullptr);
                cuDevicePrimaryCtxRelease(m_cuDevice);
                m_cuContext = nullptr;
            }
            m_cudaContextReady = false;
        }

        bool SelectCudaDeviceForVulkan() {
            VkPhysicalDeviceIDProperties idProps{};
            idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &idProps;
            vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

            int cudaDeviceCount = 0;
            if (cuDeviceGetCount(&cudaDeviceCount) != CUDA_SUCCESS || cudaDeviceCount <= 0) {
                m_lastPipelineFailureReason = "No CUDA devices available";
                return false;
            }

            for (int i = 0; i < cudaDeviceCount; ++i) {
                CUdevice candidate{};
                if (cuDeviceGet(&candidate, i) != CUDA_SUCCESS) {
                    continue;
                }

                CUuuid cuUuid{};
                if (cuDeviceGetUuid(&cuUuid, candidate) != CUDA_SUCCESS) {
                    continue;
                }

                if (memcmp(cuUuid.bytes, idProps.deviceUUID, VK_UUID_SIZE) == 0) {
                    m_cuDevice = candidate;
                    return true;
                }
            }

            m_lastPipelineFailureReason = "No CUDA device matched Vulkan physical device UUID";
            return false;
        }

        bool EnsureCudaContext() {
            if (m_cudaContextReady) {
                return true;
            }

            if (cuInit(0) != CUDA_SUCCESS) {
                m_lastPipelineFailureReason = "cuInit failed";
                return false;
            }

            if (!SelectCudaDeviceForVulkan()) {
                return false;
            }
            if (cuDevicePrimaryCtxRetain(&m_cuContext, m_cuDevice) != CUDA_SUCCESS) {
                m_lastPipelineFailureReason = "cuDevicePrimaryCtxRetain failed";
                return false;
            }
            if (cuCtxSetCurrent(m_cuContext) != CUDA_SUCCESS) {
                m_lastPipelineFailureReason = "cuCtxSetCurrent failed";
                ResetCudaContext();
                return false;
            }
            if (cuStreamCreate(&m_cuStream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
                m_lastPipelineFailureReason = "cuStreamCreate failed";
                ResetCudaContext();
                return false;
            }

            m_cudaContextReady = true;
            return true;
        }

        bool EnsureLivePipeline(uint32_t width, uint32_t height) {
            if (!EnsureCudaContext() || width == 0 || height == 0) {
                return false;
            }

            if (m_livePipelineReady && m_pipelineWidth == width && m_pipelineHeight == height) {
                return true;
            }

            if (m_livePipelineInitFailed && m_pipelineWidth == width && m_pipelineHeight == height) {
                return false;
            }

            if (m_pipelineWidth != width || m_pipelineHeight != height) {
                ResetLivePipelineResources();
            }

            try {
                m_stageCurrentColor = std::make_unique<interop::SharedImage>(
                    m_device, m_physicalDevice, width, height, VK_FORMAT_R8G8B8A8_UNORM);
                m_stagePreviousColor = std::make_unique<interop::SharedImage>(
                    m_device, m_physicalDevice, width, height, VK_FORMAT_R8G8B8A8_UNORM);
                m_stageOutputColor = std::make_unique<interop::SharedImage>(
                    m_device, m_physicalDevice, width, height, VK_FORMAT_R8G8B8A8_UNORM);

                m_stageCurrentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                m_stagePreviousLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                m_stageOutputLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                m_vkToCuda = std::make_unique<interop::SharedSemaphore>(m_device);
                m_cudaToVk = std::make_unique<interop::SharedSemaphore>(m_device);

                m_ofaPipeline = std::make_unique<OFAPipeline>(m_cuContext, width, height);
                m_frameSynthesizer = std::make_unique<FrameSynthesizer>(m_cuContext, width, height, 4);
                m_holeFiller = std::make_unique<HoleFiller>(m_cuContext, width, height);

                if (cuMemAlloc(&m_grayCurrent, width * height) != CUDA_SUCCESS) {
                    throw std::runtime_error("cuMemAlloc failed for m_grayCurrent");
                }
                if (cuMemAlloc(&m_grayPrevious, width * height) != CUDA_SUCCESS) {
                    throw std::runtime_error("cuMemAlloc failed for m_grayPrevious");
                }

                m_pipelineWidth = width;
                m_pipelineHeight = height;
                m_livePipelineReady = true;
                m_livePipelineInitFailed = false;
                m_lastPipelineFailureReason.clear();
                return true;
            } catch (const std::exception& e) {
                ResetLivePipelineResources();
                m_livePipelineInitFailed = true;
                m_pipelineWidth = width;
                m_pipelineHeight = height;
                m_lastPipelineFailureReason = e.what();
                Log(fmt::format("[ERROR] EnsureLivePipeline failed: {}\n", m_lastPipelineFailureReason));
                return false;
            } catch (...) {
                ResetLivePipelineResources();
                m_livePipelineInitFailed = true;
                m_pipelineWidth = width;
                m_pipelineHeight = height;
                m_lastPipelineFailureReason = "Unknown exception while building live pipeline";
                Log("[ERROR] EnsureLivePipeline failed with unknown exception.\n");
                return false;
            }
        }

        bool Initialize(uint32_t queueFamilyIndex) {
            if (!EnsureCudaContext()) {
                Log(fmt::format("[ERROR] Failed to initialize CUDA context for VulkanFrameProcessor: {}\n", m_lastPipelineFailureReason));
            }

            // Create Command Pool for compute/graphics operations
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = queueFamilyIndex;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            CHECK_VK_LAYER(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool));

            // Allocate Command Buffer
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = kCommandBufferRingSize;
            m_commandBuffers.resize(kCommandBufferRingSize);
            CHECK_VK_LAYER(vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()));

            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            m_submitFences.resize(kCommandBufferRingSize);
            for (uint32_t i = 0; i < kCommandBufferRingSize; ++i) {
                CHECK_VK_LAYER(vkCreateFence(m_device, &fenceInfo, nullptr, &m_submitFences[i]));
            }

            // TODO: Initialize your Vulkan Compute Pipeline here
            // (Create Descriptor Sets, Pipeline Layout, Compute Pipeline for Motion Vectors + Warp)
            return true;
        }

      public:
        ~VulkanFrameProcessor() {
            if (m_device) {
                for (VkFence fence : m_submitFences) {
                    if (fence != VK_NULL_HANDLE) {
                        vkDestroyFence(m_device, fence, nullptr);
                    }
                }
                vkDestroyCommandPool(m_device, m_commandPool, nullptr);
                // TODO: Destroy your compute pipelines here
            }

            ResetLivePipelineResources();
            ResetCudaContext();
        }

        bool CopyColorImage(VkImage sourceColor, VkImage targetColor, uint32_t width, uint32_t height) {
            if (!sourceColor || !targetColor || width == 0 || height == 0) {
                return false;
            }

            const uint32_t slot = m_submitIndex % kCommandBufferRingSize;
            const VkResult fenceStatus = vkGetFenceStatus(m_device, m_submitFences[slot]);
            if (fenceStatus == VK_NOT_READY) {
                ++m_fenceBusyCount;
                return false;
            }
            if (fenceStatus != VK_SUCCESS) {
                Log(fmt::format("[ERROR] Vulkan call failed: vkGetFenceStatus (VkResult={})\n", static_cast<int>(fenceStatus)));
                return false;
            }

            m_fenceBusyCount = 0;

            CHECK_VK_LAYER(vkResetFences(m_device, 1, &m_submitFences[slot]));

            const VkCommandBuffer cmd = m_commandBuffers[slot];
            CHECK_VK_LAYER(vkResetCommandBuffer(cmd, 0));

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK_VK_LAYER(vkBeginCommandBuffer(cmd, &beginInfo));

            VkImageMemoryBarrier toTransfer[2]{};
            const bool sourceIsCudaOutput = m_stageOutputColor && sourceColor == m_stageOutputColor->vkImage();
            toTransfer[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toTransfer[0].srcAccessMask = sourceIsCudaOutput ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            toTransfer[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            toTransfer[0].oldLayout = sourceIsCudaOutput ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            toTransfer[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toTransfer[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer[0].image = sourceColor;
            toTransfer[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toTransfer[0].subresourceRange.baseMipLevel = 0;
            toTransfer[0].subresourceRange.levelCount = 1;
            toTransfer[0].subresourceRange.baseArrayLayer = 0;
            toTransfer[0].subresourceRange.layerCount = 1;

            toTransfer[1] = toTransfer[0];
            toTransfer[1].srcAccessMask = 0;
            toTransfer[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            // The injection image has just been acquired for this frame and we overwrite it fully,
            // so we can transition from UNDEFINED and discard any previous contents.
            toTransfer[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toTransfer[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toTransfer[1].image = targetColor;

            // Source and target images have different producer stages:
            // source may come from color attachment writes, while target starts from UNDEFINED.
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr,
                                 2,
                                 toTransfer);

            VkImageCopy copyRegion{};
            copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.srcSubresource.baseArrayLayer = 0;
            copyRegion.srcSubresource.layerCount = 1;
            copyRegion.srcSubresource.mipLevel = 0;
            copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.dstSubresource.baseArrayLayer = 0;
            copyRegion.dstSubresource.layerCount = 1;
            copyRegion.dstSubresource.mipLevel = 0;
            copyRegion.extent.width = width;
            copyRegion.extent.height = height;
            copyRegion.extent.depth = 1;
            vkCmdCopyImage(cmd,
                           sourceColor,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           targetColor,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &copyRegion);

            VkImageMemoryBarrier toCompositor[2]{};
            toCompositor[0] = toTransfer[0];
            toCompositor[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            toCompositor[0].dstAccessMask = sourceIsCudaOutput ? VK_ACCESS_SHADER_READ_BIT : (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
            toCompositor[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toCompositor[0].newLayout = sourceIsCudaOutput ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            toCompositor[1] = toTransfer[1];
            toCompositor[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toCompositor[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            toCompositor[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toCompositor[1].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr,
                                 2,
                                 toCompositor);

            CHECK_VK_LAYER(vkEndCommandBuffer(cmd));

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSemaphore waitSemaphore = VK_NULL_HANDLE;
            if (sourceIsCudaOutput && m_pendingCudaOutput && m_cudaToVk) {
                waitSemaphore = m_cudaToVk->vkSemaphore();
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = &waitSemaphore;
                submitInfo.pWaitDstStageMask = &waitStage;
            }

            CHECK_VK_LAYER(vkQueueSubmit(m_queue, 1, &submitInfo, m_submitFences[slot]));
            if (sourceIsCudaOutput) {
                m_pendingCudaOutput = false;
            }

            const uint64_t frameId = ++m_frameIdCounter;
            m_slotWork[slot].submitted = true;
            m_slotWork[slot].frameId = frameId;
            m_slotWork[slot].validity = StageValidity{
                true,  // preWarp
                true,  // ofa/copy stage treated as valid for transfer path
                true,  // depthLinearization
                true,  // stereoAdaptation
                true,  // synthesis
                true   // holeFill
            };
            m_slotWork[slot].output = sourceColor;

            ++m_submitIndex;

            return true;
        }

        // ASYNCHRONOUS processing. No vkQueueWaitIdle!
        bool ProcessFrames(
            VkImage colorCurrent, VkImage depthCurrent, VkImage colorPrevious, VkImage depthPrevious, uint32_t width, uint32_t height) {
            PollCompletedWork();

            if (!colorPrevious || !depthPrevious)
                return false; // Need two frames to do motion vectors

            StageValidity stageValidity{};
            stageValidity.preWarp = colorPrevious != VK_NULL_HANDLE;

            if (!EnsureLivePipeline(width, height) || colorCurrent == VK_NULL_HANDLE) {
                m_latestStageValidity = stageValidity;
                m_latestOutputColor = VK_NULL_HANDLE;
                return false;
            }

            stageValidity.depthLinearization = depthCurrent != VK_NULL_HANDLE && depthPrevious != VK_NULL_HANDLE;
            stageValidity.stereoAdaptation = stageValidity.depthLinearization;

            const uint32_t slot = m_submitIndex % kCommandBufferRingSize;
            const VkResult fenceStatus = vkGetFenceStatus(m_device, m_submitFences[slot]);
            if (fenceStatus == VK_NOT_READY) {
                ++m_fenceBusyCount;
                return false;
            }
            if (fenceStatus != VK_SUCCESS) {
                Log(fmt::format("[ERROR] Vulkan call failed: vkGetFenceStatus (VkResult={})\n", static_cast<int>(fenceStatus)));
                return false;
            }

            m_fenceBusyCount = 0;

            CHECK_VK_LAYER(vkResetFences(m_device, 1, &m_submitFences[slot]));

            const VkCommandBuffer cmd = m_commandBuffers[slot];
            CHECK_VK_LAYER(vkResetCommandBuffer(cmd, 0));

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK_VK_LAYER(vkBeginCommandBuffer(cmd, &beginInfo));

            auto transitionImage = [&](VkImage image,
                                       VkImageLayout oldLayout,
                                       VkImageLayout newLayout,
                                       VkAccessFlags srcAccess,
                                       VkAccessFlags dstAccess,
                                       VkPipelineStageFlags srcStage,
                                       VkPipelineStageFlags dstStage) {
                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.srcAccessMask = srcAccess;
                barrier.dstAccessMask = dstAccess;
                barrier.oldLayout = oldLayout;
                barrier.newLayout = newLayout;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image;
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.baseMipLevel = 0;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.baseArrayLayer = 0;
                barrier.subresourceRange.layerCount = 1;
                vkCmdPipelineBarrier(cmd,
                                     srcStage,
                                     dstStage,
                                     0,
                                     0,
                                     nullptr,
                                     0,
                                     nullptr,
                                     1,
                                     &barrier);
            };

            auto transitionTracked = [&](VkImage image,
                                         VkImageLayout& trackedLayout,
                                         VkImageLayout newLayout,
                                         VkAccessFlags srcAccess,
                                         VkAccessFlags dstAccess,
                                         VkPipelineStageFlags srcStage,
                                         VkPipelineStageFlags dstStage) {
                transitionImage(image, trackedLayout, newLayout, srcAccess, dstAccess, srcStage, dstStage);
                trackedLayout = newLayout;
            };

            transitionImage(colorCurrent,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_ACCESS_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT);

            transitionImage(colorPrevious,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_ACCESS_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT);

            transitionTracked(m_stageCurrentColor->vkImage(),
                              m_stageCurrentLayout,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              0,
                              VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT);

            transitionTracked(m_stagePreviousColor->vkImage(),
                              m_stagePreviousLayout,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              0,
                              VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT);

            VkImageCopy copyRegion{};
            copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.srcSubresource.mipLevel = 0;
            copyRegion.srcSubresource.baseArrayLayer = 0;
            copyRegion.srcSubresource.layerCount = 1;
            copyRegion.dstSubresource = copyRegion.srcSubresource;
            copyRegion.extent.width = width;
            copyRegion.extent.height = height;
            copyRegion.extent.depth = 1;

            vkCmdCopyImage(cmd,
                           colorCurrent,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_stageCurrentColor->vkImage(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &copyRegion);

            vkCmdCopyImage(cmd,
                           colorPrevious,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_stagePreviousColor->vkImage(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &copyRegion);

            transitionImage(colorCurrent,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_ACCESS_TRANSFER_READ_BIT,
                            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

            transitionImage(colorPrevious,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_ACCESS_TRANSFER_READ_BIT,
                            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

            transitionTracked(m_stageCurrentColor->vkImage(),
                              m_stageCurrentLayout,
                              VK_IMAGE_LAYOUT_GENERAL,
                              VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_ACCESS_SHADER_READ_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            transitionTracked(m_stagePreviousColor->vkImage(),
                              m_stagePreviousLayout,
                              VK_IMAGE_LAYOUT_GENERAL,
                              VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_ACCESS_SHADER_READ_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            transitionTracked(m_stageOutputColor->vkImage(),
                              m_stageOutputLayout,
                              VK_IMAGE_LAYOUT_GENERAL,
                              0,
                              VK_ACCESS_SHADER_WRITE_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            CHECK_VK_LAYER(vkEndCommandBuffer(cmd));

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSemaphore waitSemaphore = VK_NULL_HANDLE;
            if (m_pendingCudaOutput && m_cudaToVk) {
                waitSemaphore = m_cudaToVk->vkSemaphore();
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = &waitSemaphore;
                submitInfo.pWaitDstStageMask = &waitStage;
                m_pendingCudaOutput = false;
            }

            VkSemaphore vkToCudaSemaphore = m_vkToCuda->vkSemaphore();
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &vkToCudaSemaphore;
            CHECK_VK_LAYER(vkQueueSubmit(m_queue, 1, &submitInfo, m_submitFences[slot]));

            m_vkToCuda->wait(m_cuStream);
            stageValidity.ofa = launch_rgba_to_gray(m_stageCurrentColor->cuArray(), m_grayCurrent, width, height, width, m_cuStream) &&
                                launch_rgba_to_gray(m_stagePreviousColor->cuArray(), m_grayPrevious, width, height, width, m_cuStream);

            if (stageValidity.ofa) {
                m_ofaPipeline->loadFrameDevice(0, m_grayCurrent, width, m_cuStream);
                m_ofaPipeline->loadFrameDevice(1, m_grayPrevious, width, m_cuStream);
                m_ofaPipeline->execute(m_cuStream, false);

                m_frameSynthesizer->loadFrameN(m_stagePreviousColor->cuArray(), nullptr);
                m_frameSynthesizer->loadFrameN1(m_stageCurrentColor->cuArray(), nullptr);
                const size_t vectorCount = m_ofaPipeline->outputWidth() * m_ofaPipeline->outputHeight();
                m_frameSynthesizer->loadMotionVectorsDevice(m_ofaPipeline->outputDevicePtr(), vectorCount, m_cuStream);
                m_frameSynthesizer->execute(m_cuStream, false);
                stageValidity.synthesis = true;

                m_holeFiller->fill(m_frameSynthesizer->synthesizedFrame(), m_frameSynthesizer->holeMap(), m_cuStream, false);
                stageValidity.holeFill = true;

                if (!launch_copy_rgba_array(m_frameSynthesizer->synthesizedFrame(), m_stageOutputColor->cuArray(), width, height, m_cuStream)) {
                    stageValidity.holeFill = false;
                }
            }

            m_cudaToVk->signal(m_cuStream);
            m_pendingCudaOutput = true;

            const uint64_t frameId = ++m_frameIdCounter;
            m_slotWork[slot].submitted = true;
            m_slotWork[slot].frameId = frameId;
            m_slotWork[slot].validity = stageValidity;
            m_slotWork[slot].output = stageValidity.FullyValid() ? m_stageOutputColor->vkImage() : VK_NULL_HANDLE;

            ++m_submitIndex;

            return stageValidity.FullyValid();
        }

      private:
        VkPhysicalDevice m_physicalDevice;
        VkDevice m_device;
        VkQueue m_queue;
        VkCommandPool m_commandPool;
        std::vector<VkCommandBuffer> m_commandBuffers;
        std::vector<VkFence> m_submitFences;
        uint32_t m_submitIndex{0};
        bool m_valid{false};
        StageValidity m_latestStageValidity{};
        VkImage m_latestOutputColor{VK_NULL_HANDLE};
        uint32_t m_fenceBusyCount{0};
        struct SlotWork {
            bool submitted{false};
            uint64_t frameId{0};
            StageValidity validity{};
            VkImage output{VK_NULL_HANDLE};
        };
        std::array<SlotWork, kCommandBufferRingSize> m_slotWork{};
        uint64_t m_frameIdCounter{0};
        uint64_t m_lastCompletedFrameId{0};
        bool m_livePipelineReady{false};
        bool m_livePipelineInitFailed{false};
        bool m_cudaContextReady{false};
        bool m_pendingCudaOutput{false};
        uint32_t m_pipelineWidth{0};
        uint32_t m_pipelineHeight{0};
        VkImageLayout m_stageCurrentLayout{VK_IMAGE_LAYOUT_UNDEFINED};
        VkImageLayout m_stagePreviousLayout{VK_IMAGE_LAYOUT_UNDEFINED};
        VkImageLayout m_stageOutputLayout{VK_IMAGE_LAYOUT_UNDEFINED};
        std::string m_lastPipelineFailureReason{};
        CUdevice m_cuDevice{};
        CUcontext m_cuContext{nullptr};
        CUstream m_cuStream{nullptr};
        CUdeviceptr m_grayCurrent{0};
        CUdeviceptr m_grayPrevious{0};
        std::unique_ptr<interop::SharedImage> m_stageCurrentColor;
        std::unique_ptr<interop::SharedImage> m_stagePreviousColor;
        std::unique_ptr<interop::SharedImage> m_stageOutputColor;
        std::unique_ptr<interop::SharedSemaphore> m_vkToCuda;
        std::unique_ptr<interop::SharedSemaphore> m_cudaToVk;
        std::unique_ptr<OFAPipeline> m_ofaPipeline;
        std::unique_ptr<FrameSynthesizer> m_frameSynthesizer;
        std::unique_ptr<HoleFiller> m_holeFiller;
        // TODO: Add VkPipeline, VkPipelineLayout, VkDescriptorSet etc.
    };

    // This class implements our API layer.
    class OpenXrLayer : public openxr_api_layer::OpenXrApi {
      public:
        OpenXrLayer() = default;
        ~OpenXrLayer() = default;

        XrResult xrGetVulkanInstanceExtensionsKHR(XrInstance instance,
                                                  XrSystemId systemId,
                                                  uint32_t bufferCapacityInput,
                                                  uint32_t* bufferCountOutput,
                                                  char* buffer) {
            ResolveVulkanExtensionFunctions(instance);
            if (m_xrGetVulkanInstanceExtensionsKHR == nullptr) {
                return XR_ERROR_FUNCTION_UNSUPPORTED;
            }
            return m_xrGetVulkanInstanceExtensionsKHR(instance, systemId, bufferCapacityInput, bufferCountOutput, buffer);
        }

        XrResult xrGetVulkanDeviceExtensionsKHR(XrInstance instance,
                                                XrSystemId systemId,
                                                uint32_t bufferCapacityInput,
                                                uint32_t* bufferCountOutput,
                                                char* buffer) {
            ResolveVulkanExtensionFunctions(instance);
            if (m_xrGetVulkanDeviceExtensionsKHR == nullptr) {
                return XR_ERROR_FUNCTION_UNSUPPORTED;
            }

            uint32_t runtimeCount = 0;
            const XrResult queryResult = m_xrGetVulkanDeviceExtensionsKHR(instance, systemId, 0, &runtimeCount, nullptr);
            if (XR_FAILED(queryResult)) {
                return queryResult;
            }

            std::string runtimeExtensions;
            runtimeExtensions.resize(runtimeCount == 0 ? 0 : runtimeCount - 1);
            if (runtimeCount > 0) {
                std::vector<char> runtimeBuffer(runtimeCount);
                const XrResult fetchResult =
                    m_xrGetVulkanDeviceExtensionsKHR(instance, systemId, runtimeCount, &runtimeCount, runtimeBuffer.data());
                if (XR_FAILED(fetchResult)) {
                    return fetchResult;
                }
                runtimeExtensions.assign(runtimeBuffer.data());
            }

            static const std::array<const char*, 4> requiredDeviceExtensions = {
                "VK_KHR_external_memory",
                "VK_KHR_external_memory_win32",
                "VK_KHR_external_semaphore",
                "VK_KHR_external_semaphore_win32",
            };

            std::istringstream extStream(runtimeExtensions);
            std::vector<std::string> extensions;
            std::string token;
            while (extStream >> token) {
                extensions.push_back(token);
            }

            auto hasExtension = [&](const char* extension) {
                return std::find(extensions.begin(), extensions.end(), extension) != extensions.end();
            };

            for (const char* extension : requiredDeviceExtensions) {
                if (!hasExtension(extension)) {
                    extensions.emplace_back(extension);
                }
            }

            std::string mergedExtensions;
            for (size_t i = 0; i < extensions.size(); ++i) {
                if (i != 0) {
                    mergedExtensions.push_back(' ');
                }
                mergedExtensions += extensions[i];
            }

            const uint32_t mergedCount = static_cast<uint32_t>(mergedExtensions.size()) + 1;
            if (bufferCountOutput != nullptr) {
                *bufferCountOutput = mergedCount;
            }

            TraceLoggingWrite(g_traceProvider,
                              "Vulkan_Device_Extensions",
                              TLArg(runtimeExtensions.c_str(), "RuntimeExtensions"),
                              TLArg(mergedExtensions.c_str(), "MergedExtensions"));

            if (bufferCapacityInput == 0 || buffer == nullptr) {
                return XR_SUCCESS;
            }

            if (bufferCapacityInput < mergedCount) {
                return XR_ERROR_SIZE_INSUFFICIENT;
            }

            strcpy_s(buffer, bufferCapacityInput, mergedExtensions.c_str());
            return XR_SUCCESS;
        }

        // https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrGetInstanceProcAddr
        XrResult xrGetInstanceProcAddr(XrInstance instance, const char* name, PFN_xrVoidFunction* function) override {
            TraceLoggingWrite(g_traceProvider,
                              "xrGetInstanceProcAddr",
                              TLXArg(instance, "Instance"),
                              TLArg(name, "Name"),
                              TLArg(m_bypassApiLayer, "Bypass"));

            if (!m_bypassApiLayer && name != nullptr && function != nullptr) {
                if (strcmp(name, "xrGetVulkanInstanceExtensionsKHR") == 0) {
                    *function = reinterpret_cast<PFN_xrVoidFunction>(xrGetVulkanInstanceExtensionsKHR_intercept);
                    return XR_SUCCESS;
                }
                if (strcmp(name, "xrGetVulkanDeviceExtensionsKHR") == 0) {
                    *function = reinterpret_cast<PFN_xrVoidFunction>(xrGetVulkanDeviceExtensionsKHR_intercept);
                    return XR_SUCCESS;
                }
                if (strcmp(name, "xrBeginFrame") == 0) {
                    *function = reinterpret_cast<PFN_xrVoidFunction>(xrBeginFrame_intercept);
                    return XR_SUCCESS;
                }
            }

            XrResult result = m_bypassApiLayer ? m_xrGetInstanceProcAddr(instance, name, function)
                                               : OpenXrApi::xrGetInstanceProcAddr(instance, name, function);

            TraceLoggingWrite(g_traceProvider, "xrGetInstanceProcAddr", TLPArg(*function, "Function"));

            return result;
        }

        // https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrCreateInstance
        XrResult xrCreateInstance(const XrInstanceCreateInfo* createInfo) override {
            if (createInfo->type != XR_TYPE_INSTANCE_CREATE_INFO) {
                return XR_ERROR_VALIDATION_FAILURE;
            }

            // Needed to resolve the requested function pointers.
            OpenXrApi::xrCreateInstance(createInfo);
            ResolveVulkanExtensionFunctions(GetXrInstance());

            // Dump the application name, OpenXR runtime information and other useful things for debugging.
            TraceLoggingWrite(g_traceProvider,
                              "xrCreateInstance",
                              TLArg(xr::ToString(createInfo->applicationInfo.apiVersion).c_str(), "ApiVersion"),
                              TLArg(createInfo->applicationInfo.applicationName, "ApplicationName"),
                              TLArg(createInfo->applicationInfo.applicationVersion, "ApplicationVersion"),
                              TLArg(createInfo->applicationInfo.engineName, "EngineName"),
                              TLArg(createInfo->applicationInfo.engineVersion, "EngineVersion"),
                              TLArg(createInfo->createFlags, "CreateFlags"));
            Log(fmt::format("Application: {}\n", createInfo->applicationInfo.applicationName));

            // Here there can be rules to disable the API layer entirely (based on applicationName for example).
            // m_bypassApiLayer = ...

            if (m_bypassApiLayer) {
                Log(fmt::format("{} layer will be bypassed\n", LayerName));
                return XR_SUCCESS;
            }

            for (uint32_t i = 0; i < createInfo->enabledApiLayerCount; i++) {
                TraceLoggingWrite(
                    g_traceProvider, "xrCreateInstance", TLArg(createInfo->enabledApiLayerNames[i], "ApiLayerName"));
            }
            for (uint32_t i = 0; i < createInfo->enabledExtensionCount; i++) {
                TraceLoggingWrite(
                    g_traceProvider, "xrCreateInstance", TLArg(createInfo->enabledExtensionNames[i], "ExtensionName"));
            }

            XrInstanceProperties instanceProperties = {XR_TYPE_INSTANCE_PROPERTIES};
            CHECK_XRCMD(OpenXrApi::xrGetInstanceProperties(GetXrInstance(), &instanceProperties));
            const auto runtimeName = fmt::format("{} {}.{}.{}",
                                                 instanceProperties.runtimeName,
                                                 XR_VERSION_MAJOR(instanceProperties.runtimeVersion),
                                                 XR_VERSION_MINOR(instanceProperties.runtimeVersion),
                                                 XR_VERSION_PATCH(instanceProperties.runtimeVersion));
            TraceLoggingWrite(g_traceProvider, "xrCreateInstance", TLArg(runtimeName.c_str(), "RuntimeName"));
            Log(fmt::format("Using OpenXR runtime: {}\n", runtimeName));

            return XR_SUCCESS;
        }

        // https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrGetSystem
        XrResult xrGetSystem(XrInstance instance, const XrSystemGetInfo* getInfo, XrSystemId* systemId) override {
            if (getInfo->type != XR_TYPE_SYSTEM_GET_INFO) {
                return XR_ERROR_VALIDATION_FAILURE;
            }

            TraceLoggingWrite(g_traceProvider,
                              "xrGetSystem",
                              TLXArg(instance, "Instance"),
                              TLArg(xr::ToCString(getInfo->formFactor), "FormFactor"));

            const XrResult result = OpenXrApi::xrGetSystem(instance, getInfo, systemId);
            if (XR_SUCCEEDED(result) && getInfo->formFactor == XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY) {
                if (*systemId != m_systemId) {
                    XrSystemProperties systemProperties{XR_TYPE_SYSTEM_PROPERTIES};
                    CHECK_XRCMD(OpenXrApi::xrGetSystemProperties(instance, *systemId, &systemProperties));
                    TraceLoggingWrite(g_traceProvider, "xrGetSystem", TLArg(systemProperties.systemName, "SystemName"));
                    Log(fmt::format("Using OpenXR system: {}\n", systemProperties.systemName));
                }

                // Remember the XrSystemId to use.
                m_systemId = *systemId;
            }

            TraceLoggingWrite(g_traceProvider, "xrGetSystem", TLArg((int)*systemId, "SystemId"));

            return result;
        }

        // https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrCreateSession
        // 1. HOOK SESSION CREATION
        XrResult xrCreateSession(XrInstance instance,
                                 const XrSessionCreateInfo* createInfo,
                                 XrSession* session) override {
            if (createInfo->type != XR_TYPE_SESSION_CREATE_INFO) {
                return XR_ERROR_VALIDATION_FAILURE;
            }

            // Check what engine the game is using
            const XrBaseInStructure* entry = reinterpret_cast<const XrBaseInStructure*>(createInfo->next);
            while (entry) {
                if (entry->type == XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR) {
                    Log("Star Citizen is using VULKAN\n");
                    m_apiType = GraphicsAPI::Vulkan;

                    // Capture the Vulkan handles the game is using
                    const auto* vkBinding = reinterpret_cast<const XrGraphicsBindingVulkanKHR*>(entry);
                    m_vkInstance = vkBinding->instance;
                    m_vkPhysicalDevice = vkBinding->physicalDevice;
                    m_vkDevice = vkBinding->device;
                    m_vkQueueFamilyIndex = vkBinding->queueFamilyIndex;
                    m_vkQueueIndex = vkBinding->queueIndex;
                    m_vkRuntimeQueueFamilyIndex = m_vkQueueFamilyIndex;
                    m_vkRuntimeQueueIndex = m_vkQueueIndex;

                    VkQueue appQueue = VK_NULL_HANDLE;
                    vkGetDeviceQueue(m_vkDevice, m_vkQueueFamilyIndex, m_vkQueueIndex, &appQueue);
                    m_vkAppQueue = appQueue;
                    m_vkRuntimeQueue = appQueue;

                    // Step A: probe queueIndex + 1 first (zero cost).
                    VkQueue probeQueue = VK_NULL_HANDLE;
                    vkGetDeviceQueue(m_vkDevice, m_vkQueueFamilyIndex, m_vkQueueIndex + 1, &probeQueue);
                    if (probeQueue != VK_NULL_HANDLE && probeQueue != appQueue) {
                        m_vkRuntimeQueue = probeQueue;
                        m_vkRuntimeQueueFamilyIndex = m_vkQueueFamilyIndex;
                        m_vkRuntimeQueueIndex = m_vkQueueIndex + 1;
                        m_queueIsolationSource = "probe";
                    } else if (m_enable2QueueRewriteApplied &&
                               m_enable2QueueRewriteFamily == m_vkQueueFamilyIndex) {
                        // Step B: if xrCreateVulkanDeviceKHR rewrite succeeded for this
                        // family, prefer queueIndex + 1 for the runtime thread.
                        VkQueue rewrittenQueue = VK_NULL_HANDLE;
                        vkGetDeviceQueue(m_vkDevice, m_vkQueueFamilyIndex, m_vkQueueIndex + 1, &rewrittenQueue);
                        if (rewrittenQueue != VK_NULL_HANDLE && rewrittenQueue != appQueue) {
                            m_vkRuntimeQueue = rewrittenQueue;
                            m_vkRuntimeQueueFamilyIndex = m_vkQueueFamilyIndex;
                            m_vkRuntimeQueueIndex = m_vkQueueIndex + 1;
                            m_queueIsolationSource = "enable2_rewrite";
                        } else {
                            m_queueIsolationSource = "none";
                        }
                    } else {
                        m_queueIsolationSource = "none";
                    }

                    Log(fmt::format(
                        "Queue isolation result: source={} | appQueue={} (family={}, index={}) | rtQueue={} (family={}, index={})\n",
                        m_queueIsolationSource,
                        reinterpret_cast<void*>(m_vkAppQueue),
                        m_vkQueueFamilyIndex,
                        m_vkQueueIndex,
                        reinterpret_cast<void*>(m_vkRuntimeQueue),
                        m_vkRuntimeQueueFamilyIndex,
                        m_vkRuntimeQueueIndex));
                } else if (entry->type == XR_TYPE_GRAPHICS_BINDING_D3D11_KHR) {
                    Log("Game is using DX11\n");
                    m_apiType = GraphicsAPI::DX11;
                }
                entry = entry->next;
            }

            XrResult result = OpenXrApi::xrCreateSession(instance, createInfo, session);

            if (XR_SUCCEEDED(result) && isSystemHandled(createInfo->systemId)) {
                // If a previous session's Phase3 resources are still alive, tear them down first.
                if (m_session != XR_NULL_HANDLE) {
                    TeardownPhase3Resources();
                }
                m_session = *session;
                Log(fmt::format("Captured VR Session: {}\n", (void*)m_session));

                PFN_xrVoidFunction beginFn = nullptr;
                if (XR_SUCCEEDED(OpenXrApi::xrGetInstanceProcAddr(instance, "xrBeginFrame", &beginFn))) {
                    m_xrBeginFrame = reinterpret_cast<PFN_xrBeginFrame>(beginFn);
                }

                PFN_xrVoidFunction waitFn = nullptr;
                if (XR_SUCCEEDED(OpenXrApi::xrGetInstanceProcAddr(instance, "xrWaitSwapchainImage", &waitFn))) {
                    m_xrWaitSwapchainImage = reinterpret_cast<PFN_xrWaitSwapchainImage>(waitFn);
                }

                PFN_xrVoidFunction releaseFn = nullptr;
                if (XR_SUCCEEDED(OpenXrApi::xrGetInstanceProcAddr(instance, "xrReleaseSwapchainImage", &releaseFn))) {
                    m_xrReleaseSwapchainImage = reinterpret_cast<PFN_xrReleaseSwapchainImage>(releaseFn);
                }

                PFN_xrVoidFunction createRefSpaceFn = nullptr;
                if (XR_SUCCEEDED(OpenXrApi::xrGetInstanceProcAddr(instance, "xrCreateReferenceSpace", &createRefSpaceFn))) {
                    m_xrCreateReferenceSpace = reinterpret_cast<PFN_xrCreateReferenceSpace>(createRefSpaceFn);
                }

                PFN_xrVoidFunction destroySpaceFn = nullptr;
                if (XR_SUCCEEDED(OpenXrApi::xrGetInstanceProcAddr(instance, "xrDestroySpace", &destroySpaceFn))) {
                    m_xrDestroySpace = reinterpret_cast<PFN_xrDestroySpace>(destroySpaceFn);
                }

                const bool hasIsolatedQueue =
                    (m_vkAppQueue != VK_NULL_HANDLE &&
                     m_vkRuntimeQueue != VK_NULL_HANDLE &&
                     m_vkAppQueue != m_vkRuntimeQueue);
                m_phase3DecoupledAllowed = hasIsolatedQueue;

                if (m_phase3DecoupledAllowed) {
                    // Signal deferred construction of HoldingPen + RuntimeThread.
                    // Actual construction is deferred to first xrEndFrame once the
                    // injection swapchain and its images are available.
                    m_needHoldingPenInit = true;
                    Log(fmt::format(
                        "Phase3 capability gate: enabled (source={}) | appQueue={} | rtQueue={} | enable2Intercepted={} | enable2RewriteApplied={}\n",
                        m_queueIsolationSource,
                        reinterpret_cast<void*>(m_vkAppQueue),
                        reinterpret_cast<void*>(m_vkRuntimeQueue),
                        m_enable2DeviceIntercepted,
                        m_enable2QueueRewriteApplied));
                    TraceLoggingWrite(g_traceProvider,
                                      "Phase3_CapabilityGate",
                                      TLArg(true, "DecoupledAllowed"),
                                      TLArg(m_queueIsolationSource.c_str(), "Source"),
                                      TLArg(reinterpret_cast<uint64_t>(m_vkAppQueue), "AppQueue"),
                                      TLArg(reinterpret_cast<uint64_t>(m_vkRuntimeQueue), "RuntimeQueue"),
                                      TLArg(m_vkQueueFamilyIndex, "AppQueueFamily"),
                                      TLArg(m_vkQueueIndex, "AppQueueIndex"),
                                      TLArg(m_vkRuntimeQueueFamilyIndex, "RuntimeQueueFamily"),
                                      TLArg(m_vkRuntimeQueueIndex, "RuntimeQueueIndex"));
                } else {
                    // Safe fallback policy: passthrough mode when queue isolation
                    // is not confirmed.
                    m_needHoldingPenInit = false;
                    Log(fmt::format(
                        "Phase3 capability gate: passthrough (source={}) | appQueue={} | rtQueue={} | enable2Intercepted={} | enable2RewriteApplied={}\n",
                        m_queueIsolationSource,
                        reinterpret_cast<void*>(m_vkAppQueue),
                        reinterpret_cast<void*>(m_vkRuntimeQueue),
                        m_enable2DeviceIntercepted,
                        m_enable2QueueRewriteApplied));
                    TraceLoggingWrite(g_traceProvider,
                                      "Phase3_CapabilityGate",
                                      TLArg(false, "DecoupledAllowed"),
                                      TLArg(m_queueIsolationSource.c_str(), "Source"),
                                      TLArg(reinterpret_cast<uint64_t>(m_vkAppQueue), "AppQueue"),
                                      TLArg(reinterpret_cast<uint64_t>(m_vkRuntimeQueue), "RuntimeQueue"),
                                      TLArg(m_vkQueueFamilyIndex, "AppQueueFamily"),
                                      TLArg(m_vkQueueIndex, "AppQueueIndex"),
                                      TLArg(m_vkRuntimeQueueFamilyIndex, "RuntimeQueueFamily"),
                                      TLArg(m_vkRuntimeQueueIndex, "RuntimeQueueIndex"));
                }
            }
            return result;
        }

        XrResult xrCreateVulkanDeviceKHR(XrInstance instance,
                                         const XrVulkanDeviceCreateInfoKHR* createInfo,
                                         VkDevice* vulkanDevice,
                                         VkResult* vulkanResult) override {
            m_enable2DeviceIntercepted = true;
            Log("xrCreateVulkanDeviceKHR intercepted.\n");

            if (createInfo == nullptr || createInfo->vulkanCreateInfo == nullptr) {
                return OpenXrApi::xrCreateVulkanDeviceKHR(instance, createInfo, vulkanDevice, vulkanResult);
            }

            // Build a writable copy so we can bump queueCount for single-queue
            // families that can support at least 2 queues.
            VkDeviceCreateInfo rewrittenCreateInfo = *createInfo->vulkanCreateInfo;
            std::vector<VkDeviceQueueCreateInfo> queueInfos(
                createInfo->vulkanCreateInfo->pQueueCreateInfos,
                createInfo->vulkanCreateInfo->pQueueCreateInfos + createInfo->vulkanCreateInfo->queueCreateInfoCount);
            std::vector<std::vector<float>> priorityStorage(queueInfos.size());

            PFN_vkGetPhysicalDeviceQueueFamilyProperties getFamilyProps =
                reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                    createInfo->pfnGetInstanceProcAddr(VK_NULL_HANDLE, "vkGetPhysicalDeviceQueueFamilyProperties"));

            uint32_t familyCount = 0;
            std::vector<VkQueueFamilyProperties> familyProps;
            if (getFamilyProps != nullptr) {
                getFamilyProps(createInfo->vulkanPhysicalDevice, &familyCount, nullptr);
                familyProps.resize(familyCount);
                if (familyCount > 0) {
                    getFamilyProps(createInfo->vulkanPhysicalDevice, &familyCount, familyProps.data());
                }
            }

            bool rewroteQueueCount = false;
            for (size_t i = 0; i < queueInfos.size(); ++i) {
                auto& q = queueInfos[i];
                if (q.queueCount != 1) {
                    continue;
                }
                if (q.queueFamilyIndex >= familyProps.size() ||
                    familyProps[q.queueFamilyIndex].queueCount < 2) {
                    continue;
                }

                const float basePriority = (q.pQueuePriorities != nullptr) ? q.pQueuePriorities[0] : 1.0f;
                priorityStorage[i] = {basePriority, basePriority};
                q.queueCount = 2;
                q.pQueuePriorities = priorityStorage[i].data();
                rewroteQueueCount = true;

                m_enable2QueueRewriteFamily = q.queueFamilyIndex;
            }

            if (rewroteQueueCount) {
                rewrittenCreateInfo.pQueueCreateInfos = queueInfos.data();
                rewrittenCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
                XrVulkanDeviceCreateInfoKHR rewrittenXrCreateInfo = *createInfo;
                rewrittenXrCreateInfo.vulkanCreateInfo = &rewrittenCreateInfo;

                const XrResult result =
                    OpenXrApi::xrCreateVulkanDeviceKHR(instance, &rewrittenXrCreateInfo, vulkanDevice, vulkanResult);
                if (XR_SUCCEEDED(result) && vulkanResult != nullptr && *vulkanResult == VK_SUCCESS) {
                    m_enable2QueueRewriteApplied = true;
                    Log(fmt::format("xrCreateVulkanDeviceKHR: bumped queueCount to 2 for family {}.\n",
                                    m_enable2QueueRewriteFamily));
                } else {
                    Log(fmt::format("xrCreateVulkanDeviceKHR: queueCount rewrite attempted but device create failed (xr={}, vk={}).\n",
                                    static_cast<int>(result),
                                    (vulkanResult != nullptr) ? static_cast<int>(*vulkanResult) : 0));
                }
                return result;
            }

            Log("xrCreateVulkanDeviceKHR: no eligible single-queue family to rewrite.\n");

            return OpenXrApi::xrCreateVulkanDeviceKHR(instance, createInfo, vulkanDevice, vulkanResult);
        }

        // 2. HOOK SWAPCHAIN CREATION
        XrResult xrCreateSwapchain(XrSession session,
                                   const XrSwapchainCreateInfo* createInfo,
                                   XrSwapchain* swapchain) override {
            XrResult result = OpenXrApi::xrCreateSwapchain(session, createInfo, swapchain);

            if (FrameInjection::IsCreatingSwapchain()) {
                return result;
            }

            if (XR_SUCCEEDED(result) && session == m_session && m_apiType == GraphicsAPI::Vulkan) {
                // Get color
                if (createInfo->usageFlags & XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT) {
                    const uint32_t previousWidth = m_frameBroker.GetSwapchainWidth();
                    const uint32_t previousHeight = m_frameBroker.GetSwapchainHeight();
                    if (*swapchain != m_frameInjection.Swapchain()) {
                        m_frameBroker.RegisterSwapchain(*swapchain, *createInfo);
                    }
                    Log(fmt::format("Intercepted Color Swapchain Creation! Width: {} Height: {}\n", createInfo->width, createInfo->height));

                    // Store swapchain resolution for future stereo-adapter wiring.
                    if (previousWidth != 0 && previousHeight != 0 &&
                        (previousWidth != createInfo->width || previousHeight != createInfo->height)) {
                        Log(fmt::format("[WARN] Color swapchain resolution changed from {}x{} to {}x{}; "
                            "future stereo adaptation must use per-swapchain dimensions\n",
                            previousWidth, previousHeight, createInfo->width, createInfo->height));
                    }

                    // Initialize the processor
                    if (!m_processor && m_vkDevice != VK_NULL_HANDLE) {
                        VkQueue queue = m_vkAppQueue;
                        if (queue == VK_NULL_HANDLE) {
                            vkGetDeviceQueue(m_vkDevice, m_vkQueueFamilyIndex, m_vkQueueIndex, &queue);
                        }
                        auto processor = std::make_unique<VulkanFrameProcessor>(
                            m_vkPhysicalDevice, m_vkDevice, queue, m_vkQueueFamilyIndex);
                        if (processor->IsValid()) {
                            m_processor = std::move(processor);
                            Log("Vulkan Frame Processor Initialized!\n");
                        } else {
                            Log("[ERROR] Failed to initialize Vulkan Frame Processor.\n");
                        }
                    }

                // Get depth
                } else if (createInfo->usageFlags & XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                    m_frameBroker.RegisterSwapchain(*swapchain, *createInfo);
                    Log(fmt::format("Intercepted Depth Swapchain! Width: {} Height: {}\n", createInfo->width, createInfo->height));
                }
            }
            return result;
        }

        // 3. HOOK SWAPCHAIN IMAGES (The actual GPU Textures)
        XrResult xrEnumerateSwapchainImages(XrSwapchain swapchain,
                                            uint32_t imageCapacityInput,
                                            uint32_t* imageCountOutput,
                                            XrSwapchainImageBaseHeader* images) override {
            XrResult result =
                OpenXrApi::xrEnumerateSwapchainImages(swapchain, imageCapacityInput, imageCountOutput, images);

            if (XR_SUCCEEDED(result) && m_apiType == GraphicsAPI::Vulkan) {
                const bool isPrimaryColor = m_frameBroker.IsColorSwapchain(swapchain) &&
                                            swapchain == m_frameBroker.GetPrimaryColorSwapchain();

                if (isPrimaryColor && images != nullptr) {
                    if (!m_phase3DecoupledAllowed) {
                        // Capability gate fallback: do not substitute private images
                        // when decoupled mode is disabled.
                        m_frameBroker.RegisterSwapchainImages(swapchain, *imageCountOutput, images);
                        Log(fmt::format("Mapped {} Vulkan images for primary swapchain (passthrough).\n",
                                        *imageCountOutput));
                        return result;
                    }
                    // Phase 3: return layer-owned private images to the app so SteamVR's
                    // real swapchain images are never acquired/released from the app thread.
                    // This breaks the cycle where SteamVR holds released images indefinitely
                    // (because our xrEndFrame never forwards), causing xrWaitSwapchainImage
                    // to block ~1s then return XR_ERROR_SESSION_LOST.
                    const uint32_t count = *imageCountOutput;
                    if (m_privateColorImages.empty() && m_vkDevice != VK_NULL_HANDLE) {
                        const auto ci = m_frameBroker.GetPrimaryColorCreateInfo();
                        const uint32_t w   = ci ? ci->width  : m_frameBroker.GetSwapchainWidth();
                        const uint32_t h   = ci ? ci->height : m_frameBroker.GetSwapchainHeight();
                        const VkFormat fmt = ci ? static_cast<VkFormat>(ci->format)
                                               : VK_FORMAT_R8G8B8A8_SRGB;
                        if (!AllocatePrivateColorImages(count, fmt, w, h)) {
                            Log("[WARN] Private image alloc failed — falling back to real images.\n");
                            m_frameBroker.RegisterSwapchainImages(swapchain, count, images);
                            Log(fmt::format("Mapped {} real Vulkan images for primary swapchain.\n", count));
                            return result;
                        }
                    }
                    // Overwrite the XrSwapchainImageVulkanKHR array the app supplied with
                    // our private VkImages so the app renders into layer-owned memory.
                    auto* vkImages = reinterpret_cast<XrSwapchainImageVulkanKHR*>(images);
                    for (uint32_t i = 0; i < count && i < m_privateColorImages.size(); ++i) {
                        vkImages[i].image = m_privateColorImages[i];
                    }
                    // Register private images in FrameBroker (used by xrReleaseSwapchainImage).
                    m_frameBroker.RegisterSwapchainImages(swapchain, count, images);
                    Log(fmt::format("Primary color swapchain: returned {} PRIVATE layer-owned images.\n",
                                    count));
                } else if (images != nullptr) {
                    if (m_frameBroker.IsColorSwapchain(swapchain) ||
                        m_frameBroker.IsDepthSwapchain(swapchain)) {
                        m_frameBroker.RegisterSwapchainImages(swapchain, *imageCountOutput, images);
                        Log(fmt::format("Mapped {} Vulkan images for swapchain.\n", *imageCountOutput));
                    } else if (swapchain == m_frameInjection.Swapchain()) {
                        const auto* vkImages = reinterpret_cast<const XrSwapchainImageVulkanKHR*>(images);
                        m_injectionVulkanImages.clear();
                        m_injectionVulkanImages.reserve(*imageCountOutput);
                        for (uint32_t i = 0; i < *imageCountOutput; ++i) {
                            m_injectionVulkanImages.push_back(vkImages[i].image);
                        }
                    }
                }
            }
            return result;
        }

        // HOOK ACQUIRE IMAGE (Tells us which texture index in the swapchain is active)
        XrResult xrAcquireSwapchainImage(XrSwapchain swapchain,
                                         const XrSwapchainImageAcquireInfo* acquireInfo,
                                         uint32_t* index) override {
            // RuntimeThread calls this on the synthetic swapchain — bypass layer logic
            // to prevent re-entrant frame capture that would deadlock or corrupt state.
            if (g_isRuntimeThread) {
                return OpenXrApi::xrAcquireSwapchainImage(swapchain, acquireInfo, index);
            }
            // Private color swapchain: synthesize acquire index, do NOT call SteamVR.
            // SteamVR has no knowledge of this swapchain being acquired, so it will
            // never block waiting for an xrEndFrame to recycle the image.
            if (!m_privateColorImages.empty() &&
                m_phase3DecoupledAllowed &&
                m_frameBroker.IsColorSwapchain(swapchain) &&
                swapchain == m_frameBroker.GetPrimaryColorSwapchain()) {
                *index = m_privateAcquireIndex %
                         static_cast<uint32_t>(m_privateColorImages.size());
                ++m_privateAcquireIndex;
                m_frameBroker.OnAcquireSwapchainImage(swapchain, *index);
                return XR_SUCCESS;
            }
            XrResult result = OpenXrApi::xrAcquireSwapchainImage(swapchain, acquireInfo, index);
            if (XR_SUCCEEDED(result)) {
                m_frameBroker.OnAcquireSwapchainImage(swapchain, *index);
            }
            return result;
        }

        // HOOK WAIT SWAPCHAIN IMAGE — private color swapchain images are always ready.
        // For the real swapchain this blocks until SteamVR finishes reading the image
        // from the previous frame.  For our layer-owned images there is no compositor
        // involvement, so return XR_SUCCESS immediately.
        XrResult xrWaitSwapchainImage(XrSwapchain swapchain,
                                      const XrSwapchainImageWaitInfo* waitInfo) override {
            if (g_isRuntimeThread) {
                return OpenXrApi::xrWaitSwapchainImage(swapchain, waitInfo);
            }
            if (!m_privateColorImages.empty() &&
                m_phase3DecoupledAllowed &&
                m_frameBroker.IsColorSwapchain(swapchain) &&
                swapchain == m_frameBroker.GetPrimaryColorSwapchain()) {
                return XR_SUCCESS;
            }
            return OpenXrApi::xrWaitSwapchainImage(swapchain, waitInfo);
        }

        // 2.6. HOOK SWAPCHAIN RELEASE — copy app image before SteamVR takes ownership
        // The app's color swapchain image is in COLOR_ATTACHMENT_OPTIMAL here, and
        // we own it. After the real xrReleaseSwapchainImage returns, SteamVR may
        // queue a GPU layout transition on its compositor queue. Submitting our copy
        // AFTER that point means our barrier declares the wrong oldLayout → device
        // lost. Doing it HERE, before the release, guarantees correct ownership and
        // layout. The GPU copy completes before SteamVR's compositor reads the image
        // (~11ms window vs microseconds for a memcpy-sized GPU blit).
        XrResult xrReleaseSwapchainImage(XrSwapchain swapchain,
                                          const XrSwapchainImageReleaseInfo* releaseInfo) override {
            const bool isPrimaryColor = !m_privateColorImages.empty() &&
                                        m_phase3DecoupledAllowed &&
                                        m_frameBroker.IsColorSwapchain(swapchain) &&
                                        swapchain == m_frameBroker.GetPrimaryColorSwapchain();

            if (!g_isRuntimeThread && isPrimaryColor) {
                // Copy into the HoldingPen if Phase 3 is active.
                if (m_holdingPen && m_holdingPenActive) {
                    VkImage colorImage = m_frameBroker.GetCurrentImageForSwapchain(swapchain);
                    if (colorImage != VK_NULL_HANDLE) {
                        XrPosef pose{};
                        pose.orientation = {0.f, 0.f, 0.f, 1.f};
                        if (m_frameContext.renderViews[0].valid) {
                            pose = m_frameContext.renderViews[0].pose;
                        }
                        const XrTime displayTime =
                            m_runtimeThread ? m_runtimeThread->GetLastDisplayTime() : 0;
                        const uint64_t appFrameId = ++m_appEnqueueFrameId;
                        const auto now = std::chrono::steady_clock::now().time_since_epoch();
                        const int64_t enqueueNs =
                            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
                        const bool enqueued = m_holdingPen->SubmitCopy(colorImage,
                                                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                                        displayTime, pose);
                        const uint64_t pending = m_holdingPen->GetPendingFrameCount();
                        const uint64_t drops = m_holdingPen->GetDropCount();
                        TraceLoggingWrite(g_traceProvider,
                                          "Phase3_AppEnqueue",
                                          TLArg(appFrameId, "AppFrameId"),
                                          TLArg(displayTime, "DisplayTime"),
                                          TLArg(enqueueNs, "EnqueueTimestampNs"),
                                          TLArg(enqueued, "Enqueued"),
                                          TLArg(pending, "PendingFrames"),
                                          TLArg(drops, "DropCount"));
                        if (!enqueued) {
                            Log(fmt::format("Phase3 drop policy action: drop-newest (appFrameId={}, pending={}, drops={})\n",
                                            appFrameId, pending, drops));
                        }
                    }
                }
                // Private swapchain: SteamVR never acquired this image, so do NOT forward
                // to xrReleaseSwapchainImage — there is nothing for SteamVR to release.
                return XR_SUCCESS;
            }
            return OpenXrApi::xrReleaseSwapchainImage(swapchain, releaseInfo);
        }

        // 2.5. HOOK SESSION DESTROY
        // Must stop the RuntimeThread before the session handle becomes invalid.
        // If we don't join here, the RuntimeThread is inside xrWaitFrame when
        // Log xrEndSession so we can tell whether hello_xr went through the
        // normal STOPPING state (xrEndSession called) vs. destructor-path exit.
        XrResult xrEndSession(XrSession session) override {
            Log(fmt::format("xrEndSession called (session={})\n", (void*)session));
            return OpenXrApi::xrEndSession(session);
        }

        // Intercept xrLocateViews + xrSyncActions to catch session-error returns
        // that cause hello_xr to throw (via CHECK_XRCMD) before any of our
        // other hooks are reached — which is why xrDestroySession appears without
        // a prior xrReleaseSwapchainImage or xrEndFrame in the log.
        XrResult xrLocateViews(XrSession session,
                               const XrViewLocateInfo* viewLocateInfo,
                               XrViewState* viewState,
                               uint32_t viewCapacityInput,
                               uint32_t* viewCountOutput,
                               XrView* views) override {
            XrResult result = OpenXrApi::xrLocateViews(
                session, viewLocateInfo, viewState, viewCapacityInput, viewCountOutput, views);
            if (result != XR_SUCCESS) {
                Log(fmt::format("xrLocateViews returned {} (session={})\n",
                                result, (void*)session));
            }
            return result;
        }

        XrResult xrSyncActions(XrSession session,
                               const XrActionsSyncInfo* syncInfo) override {
            XrResult result = OpenXrApi::xrSyncActions(session, syncInfo);
            if (result != XR_SUCCESS && result != XR_SESSION_NOT_FOCUSED) {
                Log(fmt::format("xrSyncActions returned {} (session={})\n",
                                result, (void*)session));
            }
            return result;
        }

        // SteamVR frees the internal session object → AV at session->field_0x30.
        XrResult xrDestroySession(XrSession session) override {
            if (session == m_session) {
                Log("xrDestroySession: tearing down Phase3 resources before destroy.\n");
                TeardownPhase3Resources();
                m_session = XR_NULL_HANDLE;
            }
            return OpenXrApi::xrDestroySession(session);
        }

        XrResult xrWaitFrame(XrSession session, const XrFrameWaitInfo* frameWaitInfo, XrFrameState* frameState) override {
            // RuntimeThread calls this with g_isRuntimeThread=true — go straight to compositor.
            if (g_isRuntimeThread) {
                return OpenXrApi::xrWaitFrame(session, frameWaitInfo, frameState);
            }

            if (session == m_session && frameState) {
                // Phase 3 active: RuntimeThread is the sole caller of the real compositor
                // xrWaitFrame. Block until it has called xrBeginFrame, then return a
                // synthetic XrFrameState. WaitForBeginFrame() is safe to call because
                // RuntimeThread stores lastTime BEFORE signalling the condvar.
                if (m_runtimeThread && m_holdingPenActive) {
                    m_runtimeThread->WaitForBeginFrame();

                    const int64_t lastTime = m_runtimeThread->GetLastDisplayTime();
                    const int64_t period   = m_runtimeThread->GetDisplayPeriod();

                    frameState->type                   = XR_TYPE_FRAME_STATE;
                    frameState->next                   = nullptr;
                    frameState->predictedDisplayTime   = lastTime + period;
                    frameState->predictedDisplayPeriod = period;
                    frameState->shouldRender           = XR_TRUE;
                    m_poseProvider.OnWaitFrame(*frameState);
                    return XR_SUCCESS;
                }

                // Phase 3 pending (HoldingPen not yet constructed): synthesize dummy
                // timing so the app NEVER calls the real compositor xrWaitFrame.
                // The real xrWaitFrame will be called exclusively by RuntimeThread once
                // it starts. Using a real call here would leave an orphaned compositor
                // frame slot that RuntimeThread can never close, causing a protocol
                // violation and session termination.
                if (m_needHoldingPenInit) {
                    constexpr int64_t kFallbackPeriodNs = 11'111'111LL; // ~90 Hz
                    // Keep predictedDisplayTime monotonically increasing across
                    // repeated calls; a constant value breaks apps that rely on it.
                    if (m_syntheticDisplayTime == 0)
                        m_syntheticDisplayTime = kFallbackPeriodNs * 2;
                    else
                        m_syntheticDisplayTime += kFallbackPeriodNs;
                    frameState->type                   = XR_TYPE_FRAME_STATE;
                    frameState->next                   = nullptr;
                    frameState->predictedDisplayTime   = m_syntheticDisplayTime;
                    frameState->predictedDisplayPeriod = kFallbackPeriodNs;
                    frameState->shouldRender           = XR_TRUE;
                    m_poseProvider.OnWaitFrame(*frameState);
                    return XR_SUCCESS;
                }
            }

            // Pre-Phase-3 (no session yet, or session mismatch) — real call.
            XrResult result = OpenXrApi::xrWaitFrame(session, frameWaitInfo, frameState);
            if (XR_SUCCEEDED(result) && session == m_session && frameState && frameState->type == XR_TYPE_FRAME_STATE) {
                m_poseProvider.OnWaitFrame(*frameState);
            }
            return result;
        }

        // 4. HOOK END FRAME (The trigger to copy the data)
        XrResult xrEndFrame(XrSession session, const XrFrameEndInfo* frameEndInfo) override {
            // RuntimeThread calls xrEndFrame on the compositor's behalf — bypass layer
            // logic entirely to avoid re-entrant frame capture or infinite recursion.
            if (g_isRuntimeThread) {
                return OpenXrApi::xrEndFrame(session, frameEndInfo);
            }
            if (session == m_session && m_apiType == GraphicsAPI::Vulkan) {
                m_frameContext = FrameContext{};
                const XrCompositionLayerProjection* projectionLayer = nullptr;

                // Extract per-eye FOV from projection layers
                if (frameEndInfo && frameEndInfo->layerCount > 0) {
                    for (uint32_t i = 0; i < frameEndInfo->layerCount; ++i) {
                        const XrCompositionLayerBaseHeader* layer = frameEndInfo->layers[i];
                        if (layer && layer->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION) {
                            projectionLayer =
                                reinterpret_cast<const XrCompositionLayerProjection*>(layer);
                            
                            // Extract FOV from both eyes
                            if (projectionLayer->viewCount >= 2) {
                                m_fovLeft = projectionLayer->views[0].fov;
                                m_fovRight = projectionLayer->views[1].fov;
                                for (uint32_t eye = 0; eye < 2; ++eye) {
                                    m_frameContext.renderViews[eye].fov = projectionLayer->views[eye].fov;
                                    m_frameContext.renderViews[eye].pose = projectionLayer->views[eye].pose;
                                    m_frameContext.renderViews[eye].valid = true;
                                }
                                
                                if (!m_fovInitialized) {
                                    m_fovInitialized = true;
                                    TraceLoggingWrite(g_traceProvider,
                                                      "FOV_Extracted",
                                                      TLArg(m_fovLeft.angleLeft, "LeftEye_AngleLeft"),
                                                      TLArg(m_fovLeft.angleRight, "LeftEye_AngleRight"),
                                                      TLArg(m_fovLeft.angleUp, "LeftEye_AngleUp"),
                                                      TLArg(m_fovLeft.angleDown, "LeftEye_AngleDown"),
                                                      TLArg(m_fovRight.angleLeft, "RightEye_AngleLeft"),
                                                      TLArg(m_fovRight.angleRight, "RightEye_AngleRight"),
                                                      TLArg(m_fovRight.angleUp, "RightEye_AngleUp"),
                                                      TLArg(m_fovRight.angleDown, "RightEye_AngleDown"));
                                }

                                // StereoVectorAdapter wiring is compiled in tests first.
                                // Runtime integration into the layer remains pending.
                            }
                            break; // Only process first projection layer
                        }
                    }
                }

                if (projectionLayer) {
                    if (m_enableFrameRewrite && m_phase3DecoupledAllowed) {
                        m_frameInjection.EnsureSwapchain(*this, session, m_frameBroker);
                        if (m_frameInjection.IsReady() && m_injectionVulkanImages.empty()) {
                            EnsureInjectionSwapchainImagesEnumerated();
                        }
                    }
                    m_depthProvider.SetSwapchainImageLookup(m_frameBroker.GetVulkanImages(), m_frameBroker.GetAcquiredIndices());
                    const XrResult poseResult =
                        m_poseProvider.PopulatePredictedViews(*this, session, *projectionLayer, m_frameContext);
                    if (XR_FAILED(poseResult)) {
                        TraceLoggingWrite(g_traceProvider,
                                          "PoseProvider_Failed",
                                          TLArg(xr::ToCString(poseResult), "Result"));
                    }

                    if (m_phase3DecoupledAllowed) {
                        bool appDepthChainPresent = false;
                        for (uint32_t eye = 0; eye < projectionLayer->viewCount; ++eye) {
                            if (projectionLayer->views[eye].next != nullptr) {
                                appDepthChainPresent = true;
                                break;
                            }
                        }
                        if (appDepthChainPresent && !m_phase3DepthIgnoredLogged) {
                            m_phase3DepthIgnoredLogged = true;
                            Log("Phase3A color-only mode: ignoring projection depth chains in decoupled path.\n");
                        }
                        TraceLoggingWrite(g_traceProvider,
                                          "Phase3_DepthIgnored",
                                          TLArg(appDepthChainPresent, "AppDepthChainPresent"),
                                          TLArg(true, "DecoupledColorOnly"));
                    } else {
                        m_depthProvider.ExtractDepthInfo(*projectionLayer, m_frameContext);
                        const bool hasDepthChain =
                            m_frameContext.depthViews[0].valid || m_frameContext.depthViews[1].valid;
                        if (!hasDepthChain && !m_depthWarningLogged) {
                            m_depthWarningLogged = true;
                            Log("[WARN] No XR_KHR_composition_layer_depth info on projection views; using tracked depth swapchains.\n");
                        }

                        if (hasDepthChain) {
                            const auto& depthView = m_frameContext.depthViews[0].valid
                                                      ? m_frameContext.depthViews[0]
                                                      : m_frameContext.depthViews[1];
                            TraceLoggingWrite(g_traceProvider,
                                              "Depth_Metadata",
                                              TLArg(depthView.minDepth, "MinDepth"),
                                              TLArg(depthView.maxDepth, "MaxDepth"),
                                              TLArg(depthView.nearZ, "NearZ"),
                                              TLArg(depthView.farZ, "FarZ"),
                                              TLArg(depthView.reversedZ, "ReversedZ"));
                        }
                    }
                }

                // Phase 3: the app thread never touches the real compositor.
                // Copies are submitted to the HoldingPen in xrReleaseSwapchainImage
                // (before image ownership transfers to SteamVR). RuntimeThread owns
                // all xrEndFrame submissions.
                EnsureHoldingPenAndRuntimeThread();

                if (m_holdingPen && m_holdingPenActive) {
                    TraceLoggingWrite(g_traceProvider,
                                      "HoldingPen_CopySubmitted",
                                      TLArg(true, "HasColor"),
                                      TLArg(frameEndInfo ? frameEndInfo->displayTime : 0LL, "DisplayTime"));
                    return XR_SUCCESS;
                }
            }

            // Fallback: Phase 3 not yet initialized — pass through.
            return OpenXrApi::xrEndFrame(session, frameEndInfo);
        }


        // Allocate N layer-owned color images returned to the app instead of SteamVR's
        // real swapchain images.  Must be called with a valid m_vkDevice.
        bool AllocatePrivateColorImages(uint32_t count, VkFormat fmt, uint32_t w, uint32_t h) {
            m_privateColorImages.assign(count, VK_NULL_HANDLE);
            m_privateColorMemories.assign(count, VK_NULL_HANDLE);

            // Free any partially-allocated resources on failure.
            auto cleanup = [&]() {
                for (uint32_t j = 0; j < count; ++j) {
                    if (m_privateColorImages[j] != VK_NULL_HANDLE)
                        vkDestroyImage(m_vkDevice, m_privateColorImages[j], nullptr);
                    if (m_privateColorMemories[j] != VK_NULL_HANDLE)
                        vkFreeMemory(m_vkDevice, m_privateColorMemories[j], nullptr);
                }
                m_privateColorImages.clear();
                m_privateColorMemories.clear();
            };

            VkPhysicalDeviceMemoryProperties memProps{};
            vkGetPhysicalDeviceMemoryProperties(m_vkPhysicalDevice, &memProps);

            for (uint32_t i = 0; i < count; ++i) {
                VkImageCreateInfo imageCI{};
                imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                imageCI.imageType     = VK_IMAGE_TYPE_2D;
                imageCI.format        = fmt;
                imageCI.extent        = {w, h, 1};
                imageCI.mipLevels     = 1;
                imageCI.arrayLayers   = 1;
                imageCI.samples       = VK_SAMPLE_COUNT_1_BIT;
                imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
                imageCI.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                imageCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
                imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                VkResult vr = vkCreateImage(m_vkDevice, &imageCI, nullptr,
                                             &m_privateColorImages[i]);
                if (vr != VK_SUCCESS) {
                    Log(fmt::format("[ERROR] AllocatePrivateColorImages: vkCreateImage failed "
                                    "(i={}, VkResult={})\n", i, static_cast<int>(vr)));
                    cleanup();
                    return false;
                }

                VkMemoryRequirements memReqs{};
                vkGetImageMemoryRequirements(m_vkDevice, m_privateColorImages[i], &memReqs);

                uint32_t memTypeIndex = UINT32_MAX;
                for (uint32_t t = 0; t < memProps.memoryTypeCount; ++t) {
                    if ((memReqs.memoryTypeBits & (1u << t)) &&
                        (memProps.memoryTypes[t].propertyFlags &
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                        memTypeIndex = t;
                        break;
                    }
                }
                if (memTypeIndex == UINT32_MAX) {
                    Log("[ERROR] AllocatePrivateColorImages: no device-local memory type\n");
                    cleanup();
                    return false;
                }

                VkMemoryAllocateInfo allocInfo{};
                allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                allocInfo.allocationSize  = memReqs.size;
                allocInfo.memoryTypeIndex = memTypeIndex;

                vr = vkAllocateMemory(m_vkDevice, &allocInfo, nullptr,
                                       &m_privateColorMemories[i]);
                if (vr != VK_SUCCESS) {
                    Log(fmt::format("[ERROR] AllocatePrivateColorImages: vkAllocateMemory failed "
                                    "(i={}, VkResult={})\n", i, static_cast<int>(vr)));
                    cleanup();
                    return false;
                }

                vr = vkBindImageMemory(m_vkDevice, m_privateColorImages[i],
                                        m_privateColorMemories[i], 0);
                if (vr != VK_SUCCESS) {
                    Log(fmt::format("[ERROR] AllocatePrivateColorImages: vkBindImageMemory failed "
                                    "(i={}, VkResult={})\n", i, static_cast<int>(vr)));
                    cleanup();
                    return false;
                }
            }
            Log(fmt::format("AllocatePrivateColorImages: {} private images {}x{} fmt={}\n",
                             count, w, h, static_cast<int>(fmt)));
            return true;
        }

        // Phase 3 teardown — called from xrCreateSession (re-create path) and destructor.
        void TeardownPhase3Resources() {
            Log("TeardownPhase3Resources: shutting down RuntimeThread and HoldingPen.\n");
            m_holdingPenActive = false;
            if (m_runtimeThread) {
                m_runtimeThread->RequestShutdownAndJoin();
                m_runtimeThread.reset();
            }
            if (m_holdingPen) {
                m_holdingPen->DrainAndDestroy();
                m_holdingPen.reset();
            }
            if (m_localSpace != XR_NULL_HANDLE && m_xrDestroySpace) {
                m_xrDestroySpace(m_localSpace);
                m_localSpace = XR_NULL_HANDLE;
            }
            if (m_vkDevice != VK_NULL_HANDLE) {
                const VkResult idleResult = vkDeviceWaitIdle(m_vkDevice);
                if (idleResult != VK_SUCCESS) {
                    Log(fmt::format("[WARN] TeardownPhase3Resources: vkDeviceWaitIdle returned {}\n",
                                    static_cast<int>(idleResult)));
                }
            }
            m_processor.reset();
            // Destroy private color images (must be after GPU is idle — HoldingPen
            // and runtime queue work. vkDeviceWaitIdle above guarantees this.
            if (m_vkDevice != VK_NULL_HANDLE) {
                for (uint32_t i = 0; i < static_cast<uint32_t>(m_privateColorImages.size()); ++i) {
                    if (m_privateColorImages[i] != VK_NULL_HANDLE) {
                        vkDestroyImage(m_vkDevice, m_privateColorImages[i], nullptr);
                    }
                    if (m_privateColorMemories[i] != VK_NULL_HANDLE) {
                        vkFreeMemory(m_vkDevice, m_privateColorMemories[i], nullptr);
                    }
                }
            }
            m_privateColorImages.clear();
            m_privateColorMemories.clear();
            m_privateAcquireIndex = 0;
            m_syntheticDisplayTime = 0;
            m_vkAppQueue = VK_NULL_HANDLE;
            m_vkRuntimeQueue = VK_NULL_HANDLE;
            m_vkRuntimeQueueFamilyIndex = 0;
            m_vkRuntimeQueueIndex = 0;
            m_queueIsolationSource = "none";
            m_phase3DecoupledAllowed = false;
            m_phase3DepthIgnoredLogged = false;
            m_appEnqueueFrameId = 0;
            m_session = XR_NULL_HANDLE;
            m_needHoldingPenInit = false;
        }

      private:
        // Deferred construction: called from xrEndFrame once the injection swapchain
        // and its images are enumerated. Safe to call every frame — exits immediately
        // once initialized.
        void EnsureHoldingPenAndRuntimeThread() {
            if (!m_phase3DecoupledAllowed) return;
            if (!m_needHoldingPenInit || m_holdingPen) return;
            if (!m_frameInjection.IsReady()) return;
            if (m_injectionVulkanImages.empty()) return;
            if (m_vkDevice == VK_NULL_HANDLE) return;

            const uint32_t w = m_frameBroker.GetSwapchainWidth();
            const uint32_t h = m_frameBroker.GetSwapchainHeight();
            if (w == 0 || h == 0) return;

            const auto infoOpt = m_frameBroker.GetPrimaryColorCreateInfo();
            if (!infoOpt) return;
            const VkFormat fmt = static_cast<VkFormat>(infoOpt->format);

            VkQueue appQueue = m_vkAppQueue;
            if (appQueue == VK_NULL_HANDLE) {
                vkGetDeviceQueue(m_vkDevice, m_vkQueueFamilyIndex, m_vkQueueIndex, &appQueue);
            }
            VkQueue runtimeQueue = m_vkRuntimeQueue;
            if (runtimeQueue == VK_NULL_HANDLE) {
                vkGetDeviceQueue(m_vkDevice, m_vkRuntimeQueueFamilyIndex, m_vkRuntimeQueueIndex, &runtimeQueue);
            }
            if (appQueue == VK_NULL_HANDLE || runtimeQueue == VK_NULL_HANDLE) return;

            // Create the local XrSpace used by RuntimeThread for projection layers.
            XrReferenceSpaceCreateInfo spaceCI{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
            spaceCI.referenceSpaceType     = XR_REFERENCE_SPACE_TYPE_LOCAL;
            spaceCI.poseInReferenceSpace   = {{0.f, 0.f, 0.f, 1.f}, {0.f, 0.f, 0.f}};
            if (!m_xrCreateReferenceSpace ||
                XR_FAILED(m_xrCreateReferenceSpace(m_session, &spaceCI, &m_localSpace))) {
                Log("[ERROR] EnsureHoldingPenAndRuntimeThread: xrCreateReferenceSpace failed.\n");
                return;
            }

            try {
                m_holdingPen = std::make_unique<HoldingPen>(
                    m_vkPhysicalDevice, m_vkDevice,
                    appQueue, m_vkQueueFamilyIndex, m_vkRuntimeQueueFamilyIndex,
                    w, h, fmt);

                m_runtimeThread = std::make_unique<RuntimeThread>(
                    *this, m_session,
                    *m_holdingPen,
                    m_frameInjection.Swapchain(),
                    m_injectionVulkanImages,
                    m_localSpace,
                    m_vkDevice,
                    runtimeQueue,
                    m_vkRuntimeQueueFamilyIndex,
                    w, h,
                    m_xrBeginFrame,
                    m_xrWaitSwapchainImage,
                    m_xrReleaseSwapchainImage);

                m_needHoldingPenInit = false;
                m_holdingPenActive   = true;
                Log(fmt::format(
                    "HoldingPen + RuntimeThread initialized: {}x{} fmt={}\n", w, h, (int)fmt));
                TraceLoggingWrite(g_traceProvider,
                                  "Phase3_Initialized",
                                  TLArg(w, "Width"),
                                  TLArg(h, "Height"),
                                  TLArg((int)fmt, "Format"));
            } catch (const std::exception& e) {
                Log(fmt::format("[ERROR] EnsureHoldingPenAndRuntimeThread: {}\n", e.what()));
                m_holdingPen.reset();
                m_runtimeThread.reset();
                if (m_localSpace != XR_NULL_HANDLE && m_xrDestroySpace) {
                    m_xrDestroySpace(m_localSpace);
                    m_localSpace = XR_NULL_HANDLE;
                }
            }
        }

        void EnsureInjectionSwapchainImagesEnumerated() {
            if (!m_frameInjection.IsReady()) {
                return;
            }

            uint32_t imageCount = 0;
            XrResult result = xrEnumerateSwapchainImages(m_frameInjection.Swapchain(), 0, &imageCount, nullptr);
            if (XR_FAILED(result) || imageCount == 0) {
                TraceLoggingWrite(g_traceProvider,
                                  "Injection_Enumerate_Failed",
                                  TLArg(xr::ToCString(result), "Result"),
                                  TLArg(imageCount, "ImageCount"));
                return;
            }

            std::vector<XrSwapchainImageVulkanKHR> images(imageCount, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
            result = xrEnumerateSwapchainImages(m_frameInjection.Swapchain(),
                                                imageCount,
                                                &imageCount,
                                                reinterpret_cast<XrSwapchainImageBaseHeader*>(images.data()));
            if (XR_FAILED(result)) {
                TraceLoggingWrite(g_traceProvider,
                                  "Injection_Enumerate_Failed",
                                  TLArg(xr::ToCString(result), "Result"),
                                  TLArg(imageCount, "ImageCount"));
            }
        }

        void ResolveVulkanExtensionFunctions(XrInstance instance) {
            if (instance == XR_NULL_HANDLE || m_xrGetInstanceProcAddr == nullptr) {
                return;
            }

            PFN_xrVoidFunction function = nullptr;
            if (m_xrGetVulkanInstanceExtensionsKHR == nullptr &&
                XR_SUCCEEDED(m_xrGetInstanceProcAddr(instance, "xrGetVulkanInstanceExtensionsKHR", &function))) {
                m_xrGetVulkanInstanceExtensionsKHR = reinterpret_cast<PFN_xrGetVulkanInstanceExtensionsKHR>(function);
            }

            function = nullptr;
            if (m_xrGetVulkanDeviceExtensionsKHR == nullptr &&
                XR_SUCCEEDED(m_xrGetInstanceProcAddr(instance, "xrGetVulkanDeviceExtensionsKHR", &function))) {
                m_xrGetVulkanDeviceExtensionsKHR = reinterpret_cast<PFN_xrGetVulkanDeviceExtensionsKHR>(function);
            }
        }

        bool isSystemHandled(XrSystemId systemId) const {return systemId == m_systemId;}
        enum class GraphicsAPI { Unknown, DX11, Vulkan };
        GraphicsAPI m_apiType{GraphicsAPI::Unknown};
        bool m_bypassApiLayer{false};
        XrSystemId m_systemId{XR_NULL_SYSTEM_ID};
        XrSession m_session{XR_NULL_HANDLE};

        std::unique_ptr<VulkanFrameProcessor> m_processor;
        std::unique_ptr<HoldingPen>           m_holdingPen;
        std::unique_ptr<RuntimeThread>        m_runtimeThread;
        bool     m_needHoldingPenInit{false};
        // Set to true in EnsureHoldingPenAndRuntimeThread immediately after the
        // HoldingPen + RuntimeThread are constructed. Gates xrReleaseSwapchainImage
        // copies and the synthetic xrWaitFrame RuntimeThread path.
        bool     m_holdingPenActive{false};
        XrSpace  m_localSpace{XR_NULL_HANDLE};
        PFN_xrGetVulkanInstanceExtensionsKHR m_xrGetVulkanInstanceExtensionsKHR{nullptr};
        PFN_xrGetVulkanDeviceExtensionsKHR m_xrGetVulkanDeviceExtensionsKHR{nullptr};
        FrameBroker m_frameBroker;
        FrameInjection m_frameInjection;
        bool m_enableFrameRewrite{true};
        PoseProvider m_poseProvider;
        DepthProvider m_depthProvider;
        FrameContext m_frameContext{};
        PFN_xrBeginFrame              m_xrBeginFrame{nullptr};
        PFN_xrWaitSwapchainImage      m_xrWaitSwapchainImage{nullptr};
        PFN_xrReleaseSwapchainImage   m_xrReleaseSwapchainImage{nullptr};
        PFN_xrCreateReferenceSpace    m_xrCreateReferenceSpace{nullptr};
        PFN_xrDestroySpace            m_xrDestroySpace{nullptr};
        bool    m_depthWarningLogged{false};
        int64_t m_syntheticDisplayTime{0};  // monotonic counter for fallback xrWaitFrame
        VkImage m_synthesizedColor{VK_NULL_HANDLE};
        bool m_hasSynthesisOutput{false};

        // Vulkan Tracking
        VkInstance m_vkInstance{VK_NULL_HANDLE};
        VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};
        VkDevice m_vkDevice{VK_NULL_HANDLE};
        VkQueue m_vkAppQueue{VK_NULL_HANDLE};
        VkQueue m_vkRuntimeQueue{VK_NULL_HANDLE};
        uint32_t m_vkQueueFamilyIndex{0};
        uint32_t m_vkQueueIndex{0};
        uint32_t m_vkRuntimeQueueFamilyIndex{0};
        uint32_t m_vkRuntimeQueueIndex{0};
        bool m_enable2DeviceIntercepted{false};
        bool m_enable2QueueRewriteApplied{false};
        uint32_t m_enable2QueueRewriteFamily{0};
        std::string m_queueIsolationSource{"none"};
        bool m_phase3DecoupledAllowed{false};
        bool m_phase3DepthIgnoredLogged{false};
        uint64_t m_appEnqueueFrameId{0};
        std::vector<VkImage> m_injectionVulkanImages;

        // Private color images owned by the layer — returned to the app instead of
        // SteamVR's real swapchain images so SteamVR never holds acquired images.
        // This prevents xrWaitSwapchainImage blocking (the root cause of session loss).
        std::vector<VkImage>        m_privateColorImages;
        std::vector<VkDeviceMemory> m_privateColorMemories;
        uint32_t                    m_privateAcquireIndex{0};

        // NEW: History tracking for Frame Generation/Warping
        VkImage m_prevColor{VK_NULL_HANDLE};
        VkImage m_prevDepth{VK_NULL_HANDLE};

        // Per-eye FOV data (extracted from xrEndFrame projection layers)
        XrFovf m_fovLeft{};
        XrFovf m_fovRight{};
        bool m_fovInitialized{false};

        // Placeholder: near/far planes for depth linearization (TODO: extract from projection matrix)
        // These are typical VR values; should be extracted from actual projection in future
        static constexpr float NEAR_PLANE = 0.1f;
        static constexpr float FAR_PLANE = 100.0f;
        static constexpr float IPD = 0.063f; // 63mm typical; TODO: query from runtime API

        // TODO (Item 4): Pre-OFA pose pre-warp integration
        // When Item 2 (Pose Data Pipeline) is complete:
        //   1. Add std::unique_ptr<pose_warp::PoseWarper> m_poseWarper;
        //   2. Uncomment pose warp includes at top of file
        //   3. Extract pose delta in xrEndFrame (display_pose * inverse(render_pose))
        //   4. Compute homography from pose delta and FOV (m_fovLeft/m_fovRight)
        //   5. Call m_poseWarper->warp() on m_prevColor before feeding to OFA
        // This will significantly improve OFA quality by removing camera motion from flow field.

      public:
        // Called by xrBeginFrame_intercept (free function below).
        // In Phase 3 (pending or active), the app thread's xrBeginFrame is always a
        // noop — RuntimeThread exclusively owns the real compositor xrBeginFrame.
        XrResult xrBeginFrame_app(XrSession session, const XrFrameBeginInfo* frameBeginInfo) {
            if (session == m_session && m_phase3DecoupledAllowed && (m_runtimeThread || m_needHoldingPenInit)) {
                // Phase 3 active or pending — app thread never touches the real compositor.
                return XR_SUCCESS;
            }
            // Pre-Phase-3 path: forward the app's struct (preserves any pNext chain).
            if (m_xrBeginFrame) {
                return m_xrBeginFrame(session, frameBeginInfo);
            }
            return XR_ERROR_RUNTIME_FAILURE;
        }
    };

    XrResult XRAPI_CALL xrGetVulkanInstanceExtensionsKHR_intercept(XrInstance instance,
                                                                    XrSystemId systemId,
                                                                    uint32_t bufferCapacityInput,
                                                                    uint32_t* bufferCountOutput,
                                                                    char* buffer) {
        auto* layer = dynamic_cast<OpenXrLayer*>(GetInstance());
        if (layer == nullptr) {
            return XR_ERROR_RUNTIME_FAILURE;
        }
        return layer->xrGetVulkanInstanceExtensionsKHR(instance, systemId, bufferCapacityInput, bufferCountOutput, buffer);
    }

    XrResult XRAPI_CALL xrGetVulkanDeviceExtensionsKHR_intercept(XrInstance instance,
                                                                  XrSystemId systemId,
                                                                  uint32_t bufferCapacityInput,
                                                                  uint32_t* bufferCountOutput,
                                                                  char* buffer) {
        auto* layer = dynamic_cast<OpenXrLayer*>(GetInstance());
        if (layer == nullptr) {
            return XR_ERROR_RUNTIME_FAILURE;
        }
        return layer->xrGetVulkanDeviceExtensionsKHR(instance, systemId, bufferCapacityInput, bufferCountOutput, buffer);
    }

    XrResult XRAPI_CALL xrBeginFrame_intercept(XrSession session,
                                               const XrFrameBeginInfo* frameBeginInfo) {
        auto* layer = dynamic_cast<OpenXrLayer*>(GetInstance());
        if (layer == nullptr) {
            return XR_ERROR_RUNTIME_FAILURE;
        }
        return layer->xrBeginFrame_app(session, frameBeginInfo);
    }

    // This method is required by the framework to instantiate your OpenXrApi implementation.
    OpenXrApi* GetInstance() {
        if (!g_instance) {
            g_instance = std::make_unique<OpenXrLayer>();
        }
        return g_instance.get();
    }

} // namespace openxr_api_layer

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        TraceLoggingRegister(openxr_api_layer::log::g_traceProvider);
        break;

    case DLL_PROCESS_DETACH:
        TraceLoggingUnregister(openxr_api_layer::log::g_traceProvider);
        break;

    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    }
    return TRUE;
}
