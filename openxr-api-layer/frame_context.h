#pragma once

#include <array>
#include <openxr/openxr.h>
#include <vulkan/vulkan.h>

namespace openxr_api_layer {

struct FrameContext {
    struct EyePose {
        XrPosef pose{};
        XrFovf fov{};
        bool valid{false};
    };

    struct EyeDepth {
        XrSwapchain swapchain{XR_NULL_HANDLE};
        uint32_t imageIndex{0};
        VkImage image{VK_NULL_HANDLE};
        float minDepth{0.0f};
        float maxDepth{1.0f};
        float nearZ{0.0f};
        float farZ{0.0f};
        bool reversedZ{false};
        bool valid{false};
    };

    XrTime displayTime{0};
    XrSpace projectionSpace{XR_NULL_HANDLE};
    XrViewStateFlags viewStateFlags{0};
    std::array<EyePose, 2> renderViews{};
    std::array<EyePose, 2> predictedViews{};
    std::array<EyeDepth, 2> depthViews{};
};

} // namespace openxr_api_layer
