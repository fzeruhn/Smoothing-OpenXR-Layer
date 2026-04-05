#pragma once

#include <map>
#include <openxr/openxr.h>
#include <vector>
#include <vulkan/vulkan.h>

namespace openxr_api_layer {

class FrameBroker {
  public:
    void RegisterSwapchain(XrSwapchain swapchain, const XrSwapchainCreateInfo& createInfo);
    void RegisterSwapchainImages(XrSwapchain swapchain,
                                 uint32_t imageCountOutput,
                                 const XrSwapchainImageBaseHeader* images);
    void OnAcquireSwapchainImage(XrSwapchain swapchain, uint32_t index);

    VkImage GetCurrentColorImage() const;
    VkImage GetCurrentDepthImage() const;

    bool IsColorSwapchain(XrSwapchain swapchain) const;
    bool IsDepthSwapchain(XrSwapchain swapchain) const;

    uint32_t GetSwapchainWidth() const;
    uint32_t GetSwapchainHeight() const;

    const std::map<XrSwapchain, std::vector<VkImage>>& GetVulkanImages() const;
    const std::map<XrSwapchain, uint32_t>& GetAcquiredIndices() const;

  private:
    VkImage GetCurrentImageForSwapchain(XrSwapchain swapchain) const;

    std::vector<XrSwapchain> m_colorSwapchains;
    std::vector<XrSwapchain> m_depthSwapchains;
    std::map<XrSwapchain, std::vector<VkImage>> m_vulkanImages;
    std::map<XrSwapchain, uint32_t> m_acquiredIndices;
    uint32_t m_swapchainWidth{0};
    uint32_t m_swapchainHeight{0};
};

} // namespace openxr_api_layer
