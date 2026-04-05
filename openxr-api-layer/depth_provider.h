#pragma once

#include "frame_context.h"
#include <map>
#include <vector>

namespace openxr_api_layer {

class DepthProvider {
  public:
    void SetSwapchainImageLookup(const std::map<XrSwapchain, std::vector<VkImage>>& vulkanImages,
                                 const std::map<XrSwapchain, uint32_t>& acquiredIndices);
    void ExtractDepthInfo(const XrCompositionLayerProjection& projectionLayer, FrameContext& frameContext) const;

  private:
    const XrCompositionLayerDepthInfoKHR* FindDepthInfo(const XrCompositionLayerProjectionView& view) const;

    const std::map<XrSwapchain, std::vector<VkImage>>* m_vulkanImages{nullptr};
    const std::map<XrSwapchain, uint32_t>* m_acquiredIndices{nullptr};
};

} // namespace openxr_api_layer
