#include "pch.h"
#include "frame_broker.h"

#include <algorithm>

namespace openxr_api_layer {

void FrameBroker::RegisterSwapchain(XrSwapchain swapchain, const XrSwapchainCreateInfo& createInfo) {
    if ((createInfo.usageFlags & XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT) != 0) {
        m_colorSwapchains.push_back(swapchain);
        if (!m_primaryColorCreateInfo.has_value()) {
            m_primaryColorCreateInfo = createInfo;
            m_swapchainWidth = createInfo.width;
            m_swapchainHeight = createInfo.height;
        }
    } else if ((createInfo.usageFlags & XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
        m_depthSwapchains.push_back(swapchain);
    }
}

void FrameBroker::RegisterSwapchainImages(XrSwapchain swapchain,
                                          uint32_t imageCountOutput,
                                          const XrSwapchainImageBaseHeader* images) {
    if (!images) {
        return;
    }

    const bool isTracked = IsColorSwapchain(swapchain) || IsDepthSwapchain(swapchain);
    if (!isTracked) {
        return;
    }

    std::vector<VkImage>& dst = m_vulkanImages[swapchain];
    dst.clear();
    dst.reserve(imageCountOutput);

    const auto* vkImages = reinterpret_cast<const XrSwapchainImageVulkanKHR*>(images);
    for (uint32_t i = 0; i < imageCountOutput; ++i) {
        dst.push_back(vkImages[i].image);
    }
}

void FrameBroker::OnAcquireSwapchainImage(XrSwapchain swapchain, uint32_t index) {
    m_acquiredIndices[swapchain] = index;
}

VkImage FrameBroker::GetCurrentImageForSwapchain(XrSwapchain swapchain) const {
    const auto acquiredIt = m_acquiredIndices.find(swapchain);
    const auto imagesIt = m_vulkanImages.find(swapchain);
    if (acquiredIt == m_acquiredIndices.end() || imagesIt == m_vulkanImages.end()) {
        return VK_NULL_HANDLE;
    }

    const uint32_t index = acquiredIt->second;
    if (index >= imagesIt->second.size()) {
        return VK_NULL_HANDLE;
    }

    return imagesIt->second[index];
}

VkImage FrameBroker::GetCurrentColorImage() const {
    if (m_colorSwapchains.empty()) {
        return VK_NULL_HANDLE;
    }
    return GetCurrentImageForSwapchain(m_colorSwapchains.front());
}

VkImage FrameBroker::GetCurrentDepthImage() const {
    if (m_depthSwapchains.empty()) {
        return VK_NULL_HANDLE;
    }
    return GetCurrentImageForSwapchain(m_depthSwapchains.front());
}

XrSwapchain FrameBroker::GetPrimaryColorSwapchain() const {
    if (m_colorSwapchains.empty()) {
        return XR_NULL_HANDLE;
    }
    return m_colorSwapchains.front();
}

std::optional<XrSwapchainCreateInfo> FrameBroker::GetPrimaryColorCreateInfo() const {
    return m_primaryColorCreateInfo;
}

bool FrameBroker::IsColorSwapchain(XrSwapchain swapchain) const {
    return std::find(m_colorSwapchains.begin(), m_colorSwapchains.end(), swapchain) != m_colorSwapchains.end();
}

bool FrameBroker::IsDepthSwapchain(XrSwapchain swapchain) const {
    return std::find(m_depthSwapchains.begin(), m_depthSwapchains.end(), swapchain) != m_depthSwapchains.end();
}

uint32_t FrameBroker::GetSwapchainWidth() const {
    return m_swapchainWidth;
}

uint32_t FrameBroker::GetSwapchainHeight() const {
    return m_swapchainHeight;
}

const std::map<XrSwapchain, std::vector<VkImage>>& FrameBroker::GetVulkanImages() const {
    return m_vulkanImages;
}

const std::map<XrSwapchain, uint32_t>& FrameBroker::GetAcquiredIndices() const {
    return m_acquiredIndices;
}

} // namespace openxr_api_layer
