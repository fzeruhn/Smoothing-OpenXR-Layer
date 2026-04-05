#include "pch.h"
#include "frame_broker.h"
#include "frame_injection.h"
#include "layer.h"

namespace openxr_api_layer {

thread_local bool FrameInjection::s_creatingSwapchain = false;

bool FrameInjection::IsCreatingSwapchain() {
    return s_creatingSwapchain;
}

void FrameInjection::EnsureSwapchain(OpenXrApi& api, XrSession session, const FrameBroker& broker) {
    if (m_injectionSwapchain != XR_NULL_HANDLE) {
        return;
    }

    const auto baseInfo = broker.GetPrimaryColorCreateInfo();
    if (!baseInfo.has_value()) {
        return;
    }

    XrSwapchainCreateInfo createInfo = *baseInfo;
    createInfo.usageFlags |= XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    createInfo.faceCount = 1;
    createInfo.arraySize = 1;
    createInfo.mipCount = 1;

    XrSwapchain swapchain = XR_NULL_HANDLE;
    s_creatingSwapchain = true;
    const XrResult result = api.xrCreateSwapchain(session, &createInfo, &swapchain);
    s_creatingSwapchain = false;
    if (XR_SUCCEEDED(result)) {
        m_injectionSwapchain = swapchain;
    }
}

bool FrameInjection::IsReady() const {
    return m_injectionSwapchain != XR_NULL_HANDLE;
}

XrSwapchain FrameInjection::Swapchain() const {
    return m_injectionSwapchain;
}

} // namespace openxr_api_layer
