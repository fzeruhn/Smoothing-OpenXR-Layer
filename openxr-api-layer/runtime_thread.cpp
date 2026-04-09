#include "pch.h"
#include "runtime_thread.h"
#include "layer.h"

#include <stdexcept>
#include <string>
#include <cstring>
#include <mutex>

namespace openxr_api_layer {

// Defined in layer.cpp.
extern thread_local bool g_isRuntimeThread;
extern std::mutex g_queueMutex;

#define CHECK_VK(call)                                                               \
    do {                                                                             \
        VkResult _r = (call);                                                        \
        if (_r != VK_SUCCESS)                                                        \
            throw std::runtime_error(std::string("RuntimeThread VK error ") +       \
                                     std::to_string(_r) + " in " #call);            \
    } while (0)

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

RuntimeThread::RuntimeThread(OpenXrApi& api,
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
                             PFN_xrReleaseSwapchainImage releaseFn)
    : m_api(api),
      m_session(session),
      m_holdingPen(holdingPen),
      m_injectionSwapchain(injectionSwapchain),
      m_injectionVkImages(injectionVkImages),
      m_localSpace(localSpace),
      m_device(device),
      m_runtimeQueue(runtimeQueue),
      m_runtimeQueueFamily(runtimeQueueFamily),
      m_imageWidth(imageWidth),
      m_imageHeight(imageHeight),
      m_xrBeginFrame(beginFrameFn),
      m_xrWaitSwapchainImage(waitFn),
      m_xrReleaseSwapchainImage(releaseFn) {

    // Command pool for blit submits (runtime thread queue family).
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = runtimeQueueFamily;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK_VK(vkCreateCommandPool(device, &poolCI, nullptr, &m_blitPool));

    VkCommandBufferAllocateInfo cbAI{};
    cbAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAI.commandPool        = m_blitPool;
    cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = 1;
    CHECK_VK(vkAllocateCommandBuffers(device, &cbAI, &m_blitCmd));

    // Blit-done fence — starts signaled so the first vkResetFences is valid.
    VkFenceCreateInfo blitFenceCI{};
    blitFenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    blitFenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    CHECK_VK(vkCreateFence(device, &blitFenceCI, nullptr, &m_blitDoneFence));

    // Start thread last — all state must be initialized first.
    m_thread = std::thread([this] { ThreadBody(); });
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

RuntimeThread::~RuntimeThread() {
    RequestShutdownAndJoin();
    if (m_blitDoneFence != VK_NULL_HANDLE) {
        vkDestroyFence(m_device, m_blitDoneFence, nullptr);
    }
    if (m_blitPool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_device, m_blitPool, 1, &m_blitCmd);
        vkDestroyCommandPool(m_device, m_blitPool, nullptr);
    }
}

// ---------------------------------------------------------------------------
// RequestShutdownAndJoin
// ---------------------------------------------------------------------------

void RuntimeThread::RequestShutdownAndJoin() {
    m_shutdownRequested.store(true, std::memory_order_release);
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

// ---------------------------------------------------------------------------
// ThreadBody — main pacing loop
// ---------------------------------------------------------------------------

void RuntimeThread::ThreadBody() {
    // Mark this thread so layer hooks bypass frame-capture logic.
    g_isRuntimeThread = true;

    while (!m_shutdownRequested.load(std::memory_order_acquire)) {

        // ---- xrWaitFrame ----
        XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState frameState{XR_TYPE_FRAME_STATE};
        XrResult result = m_api.xrWaitFrame(m_session, &waitInfo, &frameState);
        if (XR_FAILED(result)) {
            break;
        }

        // Publish timing for app-thread synthetic xrWaitFrame.
        if (frameState.predictedDisplayPeriod > 0)
            m_displayPeriod.store(frameState.predictedDisplayPeriod, std::memory_order_release);
        m_lastDisplayTime.store(frameState.predictedDisplayTime, std::memory_order_release);
        if (m_shutdownRequested.load(std::memory_order_acquire)) {
            // Still must call xrBeginFrame + xrEndFrame to keep the session valid.
            XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
            if (m_xrBeginFrame) m_xrBeginFrame(m_session, &beginInfo);
            // Signal before SubmitBlackFrame so the app thread doesn't deadlock
            // in WaitForBeginFrame() while we're doing the final submission.
            {
                std::lock_guard<std::mutex> lk(m_beginMutex);
                m_beginFrameReady = true;
            }
            m_beginCv.notify_one();
            SubmitBlackFrame(frameState.predictedDisplayTime);
            break;
        }

        // ---- xrBeginFrame ----
        XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
        if (m_xrBeginFrame) m_xrBeginFrame(m_session, &beginInfo);

        // Signal the app thread's synthetic xrWaitFrame that xrBeginFrame has
        // been called. This unblocks the app so it can safely call
        // xrAcquireSwapchainImage (which the OpenXR spec requires to follow
        // xrBeginFrame on the compositor side).
        {
            std::lock_guard<std::mutex> lk(m_beginMutex);
            m_beginFrameReady = true;
        }
        m_beginCv.notify_one();

        if (!frameState.shouldRender) {
            SubmitBlackFrame(frameState.predictedDisplayTime);
            continue;
        }

        // ---- Path A / Path B decision ----
        auto slot = m_holdingPen.ConsumeLatest();
        if (slot.has_value()) {
            // Path A: fresh frame available.
            m_consecutivePathBCount = 0;
            // Pass the consumed fence so it fires when the GPU finishes the blit,
            // not via a separate CPU-side empty submit. This prevents the app thread
            // from reusing the slot image while the runtime thread's blit is in-flight.
            SubmitSlotImage(frameState.predictedDisplayTime,
                            slot->image,
                            slot->copyDone,
                            slot->meta.renderPose,
                            m_holdingPen.GetConsumedFence(slot->index));
            m_lastSubmittedSlot = slot;
        } else {
            // Path B: deadline miss.
            ++m_consecutivePathBCount;

            if (m_consecutivePathBCount > 10) {
                // App appears stalled — black frame to avoid smearing.
                SubmitBlackFrame(frameState.predictedDisplayTime);
            } else if (m_lastSubmittedSlot.has_value()) {
                SubmitCachedFrame(frameState.predictedDisplayTime);
            } else {
                // No frame ever received yet.
                SubmitBlackFrame(frameState.predictedDisplayTime);
            }
        }
    }

    // Wake any app thread waiting in WaitForBeginFrame so it doesn't deadlock.
    {
        std::lock_guard<std::mutex> lk(m_beginMutex);
        m_beginFrameReady = true;
    }
    m_beginCv.notify_all();

    g_isRuntimeThread = false;
}

// ---------------------------------------------------------------------------
// SubmitSlotImage — Path A
// ---------------------------------------------------------------------------

void RuntimeThread::SubmitSlotImage(XrTime displayTime,
                                    VkImage sourceImage,
                                    VkSemaphore waitSemaphore,
                                    XrPosef pose,
                                    VkFence consumedFence) {
    if (m_injectionSwapchain == XR_NULL_HANDLE) {
        SubmitBlackFrame(displayTime);
        return;
    }

    // --- Acquire injection swapchain image ---
    XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    uint32_t imageIndex = 0;
    // g_isRuntimeThread = true, so this bypasses layer frame-capture logic.
    XrResult acquireResult = m_api.xrAcquireSwapchainImage(
        m_injectionSwapchain, &acquireInfo, &imageIndex);
    if (XR_FAILED(acquireResult)) {
        SubmitBlackFrame(displayTime);
        return;
    }

    if (m_xrWaitSwapchainImage) {
        XrSwapchainImageWaitInfo swWaitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        swWaitInfo.timeout = XR_INFINITE_DURATION;
        m_xrWaitSwapchainImage(m_injectionSwapchain, &swWaitInfo);
    }

    // --- Blit sourceImage → injection image ---
    bool blitOk = false;
    if (imageIndex < m_injectionVkImages.size() && sourceImage != VK_NULL_HANDLE) {
        VkImage injectionImage = m_injectionVkImages[imageIndex];

        vkResetCommandBuffer(m_blitCmd, 0);
        VkCommandBufferBeginInfo cbBegin{};
        cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_blitCmd, &cbBegin) == VK_SUCCESS) {

            // Barrier: injection image UNDEFINED → TRANSFER_DST.
            VkImageMemoryBarrier toDst{};
            toDst.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toDst.srcAccessMask       = 0;
            toDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            toDst.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            toDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toDst.image               = injectionImage;
            toDst.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(m_blitCmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &toDst);

            // Barrier: source image SHADER_READ_ONLY → TRANSFER_SRC.
            VkImageMemoryBarrier toSrc{};
            toSrc.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toSrc.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
            toSrc.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
            toSrc.oldLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toSrc.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.image               = sourceImage;
            toSrc.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(m_blitCmd,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &toSrc);

            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.extent         = {m_imageWidth, m_imageHeight, 1};
            vkCmdCopyImage(m_blitCmd,
                sourceImage,    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                injectionImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &region);

            // Barrier: injection image TRANSFER_DST → COLOR_ATTACHMENT_OPTIMAL.
            // dstAccessMask=0 / BOTTOM_OF_PIPE: this is a queue-release barrier.
            // The compositor acquires the image on its own queue and handles its
            // own visibility — we only need to complete the layout transition here.
            VkImageMemoryBarrier toPresent{};
            toPresent.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toPresent.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            toPresent.dstAccessMask       = 0;
            toPresent.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toPresent.newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toPresent.image               = injectionImage;
            toPresent.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(m_blitCmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0, 0, nullptr, 0, nullptr, 1, &toPresent);

            vkEndCommandBuffer(m_blitCmd);

            // Submit blit. If waitSemaphore is valid, wait for the copy to finish
            // on the GPU before we read from the holding pen image.
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo submitInfo{};
            submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers    = &m_blitCmd;
            if (waitSemaphore != VK_NULL_HANDLE) {
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores    = &waitSemaphore;
                submitInfo.pWaitDstStageMask  = &waitStage;
            }
            {
                // Choose the fence for this blit submit:
                //   Path A (consumedFence valid): signals when GPU finishes,
                //     gates both SubmitCopy slot reuse and our pre-release wait.
                //   Path B (consumedFence null / cached resubmit): use
                //     m_blitDoneFence so we still get a CPU completion signal.
                VkFence submitFence = consumedFence;
                if (submitFence == VK_NULL_HANDLE) {
                    // m_blitDoneFence is signaled from construction or previous
                    // wait+implicit-signal — safe to reset before reuse.
                    vkResetFences(m_device, 1, &m_blitDoneFence);
                    submitFence = m_blitDoneFence;
                }
                std::lock_guard<std::mutex> lock(g_queueMutex);
                blitOk = (vkQueueSubmit(m_runtimeQueue, 1, &submitInfo, submitFence) == VK_SUCCESS);
            }
        }
    }

    // Wait for the GPU to finish the blit before handing the injection image
    // to SteamVR via xrReleaseSwapchainImage. Without this wait, the compositor
    // reads a partially-written image, causing "compositor failed EndFrame()".
    if (blitOk) {
        VkFence waitFence = (consumedFence != VK_NULL_HANDLE) ? consumedFence : m_blitDoneFence;
        vkWaitForFences(m_device, 1, &waitFence, VK_TRUE, UINT64_MAX);
        // m_blitDoneFence is now signaled; leave it for the implicit reset on
        // next vkResetFences call. consumedFence remains signaled for SubmitCopy
        // to poll — it will be reset by the app thread when it reuses the slot.
    }

    if (m_xrReleaseSwapchainImage) {
        XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        m_xrReleaseSwapchainImage(m_injectionSwapchain, &releaseInfo);
    }

    if (!blitOk) {
        // Injection image not available yet — fall back to black rather than
        // submitting a frame pointing at an uninitialised image.
        SubmitBlackFrame(displayTime);
        return;
    }

    // --- Build XrFrameEndInfo pointing at the injection swapchain ---
    // Color-only: no depth pNext chain (Phase 3 stabilisation).
    XrSwapchainSubImage subImage{};
    subImage.swapchain             = m_injectionSwapchain;
    subImage.imageArrayIndex       = 0;
    subImage.imageRect.offset      = {0, 0};
    subImage.imageRect.extent      = {static_cast<int32_t>(m_imageWidth),
                                      static_cast<int32_t>(m_imageHeight)};

    // Stereo requires viewCount=2. Both eyes receive the same full-resolution
    // injection image with the same pose. Phase 4 will supply per-eye poses and
    // a warped right-eye image; for Phase 3 this satisfies the spec requirement
    // and SteamVR's strict viewCount validation without crashing.
    XrCompositionLayerProjectionView projViews[2]{};
    for (int eye = 0; eye < 2; ++eye) {
        projViews[eye].type     = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
        projViews[eye].subImage = subImage;
        projViews[eye].pose     = pose;
        // Default FOV (~90° horizontal, 90° vertical). Phase 4 will thread real FOV.
        projViews[eye].fov      = {-0.7854f, 0.7854f, 0.7854f, -0.7854f};
    }

    XrCompositionLayerProjection projLayer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    projLayer.space      = m_localSpace;
    projLayer.viewCount  = 2;
    projLayer.views      = projViews;

    const XrCompositionLayerBaseHeader* layers[] = {
        reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projLayer)
    };

    XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
    endInfo.displayTime          = displayTime;
    endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;

    // Only attach the layer if we have a valid XrSpace; otherwise submit empty.
    if (m_localSpace != XR_NULL_HANDLE) {
        endInfo.layerCount = 1;
        endInfo.layers     = layers;
    }

    // g_isRuntimeThread = true, so xrEndFrame fast-paths to the downstream dispatch.
    m_api.xrEndFrame(m_session, &endInfo);
}

// ---------------------------------------------------------------------------
// SubmitCachedFrame — Path B (Phase 3: resubmit last frame unwarped)
// ---------------------------------------------------------------------------

void RuntimeThread::SubmitCachedFrame(XrTime displayTime) {
    if (!m_lastSubmittedSlot.has_value()) {
        SubmitBlackFrame(displayTime);
        return;
    }
    // Resubmit the cached slot image without warping.
    // VK_NULL_HANDLE for waitSemaphore — the image is already in
    // SHADER_READ_ONLY layout from the previous Path A blit.
    SubmitSlotImage(displayTime,
                    m_lastSubmittedSlot->image,
                    VK_NULL_HANDLE,
                    m_lastSubmittedSlot->meta.renderPose);
}

// ---------------------------------------------------------------------------
// SubmitBlackFrame — minimal valid submission (avoids layerCount=0)
// ---------------------------------------------------------------------------

void RuntimeThread::SubmitBlackFrame(XrTime displayTime) {
    // Submitting layerCount=0 activates SteamVR's depth-reprojection codepath,
    // which has a Vulkan validation bug that causes it to throw std::system_error.
    // Use the injection swapchain with an identity pose instead — image content
    // may be uninitialized/stale, but the compositor won't crash on a non-empty
    // submission. Falls back to the empty submission only if the injection
    // swapchain is not available yet (shouldn't happen in practice).
    if (m_injectionSwapchain != XR_NULL_HANDLE && m_localSpace != XR_NULL_HANDLE) {
        XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        uint32_t imageIndex = 0;
        XrResult acquireResult = m_api.xrAcquireSwapchainImage(
            m_injectionSwapchain, &acquireInfo, &imageIndex);
        if (XR_SUCCEEDED(acquireResult)) {
            if (m_xrWaitSwapchainImage) {
                XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
                waitInfo.timeout = XR_INFINITE_DURATION;
                m_xrWaitSwapchainImage(m_injectionSwapchain, &waitInfo);
            }
            if (m_xrReleaseSwapchainImage) {
                XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                m_xrReleaseSwapchainImage(m_injectionSwapchain, &releaseInfo);
            }

            // Reuse the same projection structure as SubmitSlotImage.
            XrSwapchainSubImage subImage{};
            subImage.swapchain        = m_injectionSwapchain;
            subImage.imageArrayIndex  = 0;
            subImage.imageRect.offset = {0, 0};
            subImage.imageRect.extent = {static_cast<int32_t>(m_imageWidth),
                                         static_cast<int32_t>(m_imageHeight)};

            XrPosef identityPose{};
            identityPose.orientation = {0.f, 0.f, 0.f, 1.f};

            XrCompositionLayerProjectionView projViews[2]{};
            for (int eye = 0; eye < 2; ++eye) {
                projViews[eye].type     = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
                projViews[eye].subImage = subImage;
                projViews[eye].pose     = identityPose;
                projViews[eye].fov      = {-0.7854f, 0.7854f, 0.7854f, -0.7854f};
            }

            XrCompositionLayerProjection projLayer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
            projLayer.space     = m_localSpace;
            projLayer.viewCount = 2;
            projLayer.views     = projViews;

            const XrCompositionLayerBaseHeader* layers[] = {
                reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projLayer)
            };

            XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
            endInfo.displayTime          = displayTime;
            endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            endInfo.layerCount           = 1;
            endInfo.layers               = layers;
            m_api.xrEndFrame(m_session, &endInfo);
            return;
        }
    }

    // True fallback: injection swapchain not yet available.
    XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
    endInfo.displayTime          = displayTime;
    endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    endInfo.layerCount           = 0;
    m_api.xrEndFrame(m_session, &endInfo);
}

// ---------------------------------------------------------------------------
// StripDepthChains
// ---------------------------------------------------------------------------

void RuntimeThread::StripDepthChains(XrCompositionLayerProjectionView* views,
                                     uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        // Walk the pNext chain and remove any XrCompositionLayerDepthInfoKHR.
        // For Phase 3 we take the blunt approach: null out the whole pNext chain
        // on each view, since the only extension we'd attach is depth.
        views[i].next = nullptr;
    }
}

} // namespace openxr_api_layer
