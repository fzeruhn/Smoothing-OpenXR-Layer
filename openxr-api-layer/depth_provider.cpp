#include "pch.h"
#include "depth_provider.h"

namespace openxr_api_layer {

void DepthProvider::SetSwapchainImageLookup(const std::map<XrSwapchain, std::vector<VkImage>>& vulkanImages,
                                            const std::map<XrSwapchain, uint32_t>& acquiredIndices) {
    m_vulkanImages = &vulkanImages;
    m_acquiredIndices = &acquiredIndices;
}

const XrCompositionLayerDepthInfoKHR* DepthProvider::FindDepthInfo(const XrCompositionLayerProjectionView& view) const {
    const XrBaseInStructure* entry = reinterpret_cast<const XrBaseInStructure*>(view.next);
    while (entry) {
        if (entry->type == XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR) {
            return reinterpret_cast<const XrCompositionLayerDepthInfoKHR*>(entry);
        }
        entry = entry->next;
    }
    return nullptr;
}

void DepthProvider::ExtractDepthInfo(const XrCompositionLayerProjection& projectionLayer, FrameContext& frameContext) const {
    if (!m_vulkanImages || !m_acquiredIndices || projectionLayer.viewCount < 2) {
        return;
    }

    for (uint32_t eye = 0; eye < 2; ++eye) {
        const XrCompositionLayerDepthInfoKHR* depthInfo = FindDepthInfo(projectionLayer.views[eye]);
        if (!depthInfo || depthInfo->subImage.swapchain == XR_NULL_HANDLE) {
            continue;
        }

        auto acquiredIt = m_acquiredIndices->find(depthInfo->subImage.swapchain);
        auto imagesIt = m_vulkanImages->find(depthInfo->subImage.swapchain);
        if (acquiredIt == m_acquiredIndices->end() || imagesIt == m_vulkanImages->end()) {
            continue;
        }

        const uint32_t imageIndex = acquiredIt->second;
        if (imageIndex >= imagesIt->second.size()) {
            continue;
        }

        auto& dst = frameContext.depthViews[eye];
        dst.swapchain = depthInfo->subImage.swapchain;
        dst.imageIndex = imageIndex;
        dst.image = imagesIt->second[imageIndex];
        dst.minDepth = depthInfo->minDepth;
        dst.maxDepth = depthInfo->maxDepth;
        dst.nearZ = depthInfo->nearZ;
        dst.farZ = depthInfo->farZ;
        dst.reversedZ = depthInfo->nearZ > depthInfo->farZ;
        dst.valid = true;
    }
}

} // namespace openxr_api_layer
