#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdint>

namespace openxr_api_layer {

class OpenXrApi;
class HoldingPen;

// Owns the xrWaitFrame → xrBeginFrame → xrEndFrame loop to the compositor.
// App thread must NOT call those three functions while this thread is running.
class RuntimeThread {
  public:
    struct Config {
        OpenXrApi*              api{nullptr};
        XrSession               session{XR_NULL_HANDLE};
        // Injection swapchain (Task 6 Step C; unused in Step B)
        XrSwapchain             injectionSwapchain{XR_NULL_HANDLE};
        std::vector<VkImage>    injectionImages;
        // Vulkan resources
        VkDevice                device{VK_NULL_HANDLE};
        VkPhysicalDevice        physDevice{VK_NULL_HANDLE};
        VkQueue                 queue{VK_NULL_HANDLE};
        std::mutex*             queueMutex{nullptr};
        uint32_t                queueFamilyIndex{0};
        uint32_t                width{0};
        uint32_t                height{0};
        // Holding pen (shared with app thread)
        HoldingPen*             holdingPen{nullptr};
        // OpenXR function pointers
        PFN_xrWaitSwapchainImage    xrWaitSwapchainImage{nullptr};
        PFN_xrReleaseSwapchainImage xrReleaseSwapchainImage{nullptr};
    };

    // Start the runtime thread. Returns false if config is invalid.
    bool Start(Config config);

    // Signal thread to stop and block until it exits. Safe to call multiple times.
    void Stop();

    [[nodiscard]] bool IsRunning() const { return m_running.load(); }

    // App thread calls this to get a synthesized XrFrameState derived from the
    // runtime thread's last xrWaitFrame result. Returns false if not yet valid.
    bool GetSynthesizedFrameState(XrFrameState& frameState);

  private:
    void ThreadMain();
    bool InitVulkanResources();
    void CleanupVulkanResources();

    // Step B: submit 0-layer frame to compositor
    void SubmitEmptyFrame(XrTime displayTime);
    // Step C (Task 6): copy holding-pen slot → injection image, submit rewritten frame
    bool SubmitColorFrame(int32_t holdingSlot, XrTime displayTime);

    Config              m_config;
    std::thread         m_thread;
    std::atomic<bool>   m_stop{false};
    std::atomic<bool>   m_running{false};

    // Vulkan resources owned exclusively by the runtime thread
    VkCommandPool   m_commandPool{VK_NULL_HANDLE};
    VkCommandBuffer m_commandBuffer{VK_NULL_HANDLE};
    VkFence         m_copyFence{VK_NULL_HANDLE};

    // Synthesized frame state published to the app thread
    std::mutex  m_frameStateMutex;
    XrTime      m_lastDisplayTime{0};
    XrDuration  m_displayPeriod{11111111};  // ~90 Hz in nanoseconds; updated each frame
    bool        m_frameStateValid{false};
};

} // namespace openxr_api_layer
