#pragma once

#include <openxr/openxr.h>

namespace openxr_api_layer {

class OpenXrApi;
class FrameBroker;

class FrameInjection {
  public:
    static bool IsCreatingSwapchain();

    void EnsureSwapchain(OpenXrApi& api, XrSession session, const FrameBroker& broker);
    bool IsReady() const;
    XrSwapchain Swapchain() const;

  private:
    static thread_local bool s_creatingSwapchain;
    XrSwapchain m_injectionSwapchain{XR_NULL_HANDLE};
};

} // namespace openxr_api_layer
