#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>

#include <atomic>
#include <array>
#include <optional>
#include <cstdint>
#include <vector>

namespace openxr_api_layer {

// Manages a 3-slot ring buffer of layer-owned VkImages used as a "holding pen"
// between the app thread (which writes copies) and the runtime thread (which reads
// and submits to the compositor).
//
// Thread safety:
//   SubmitCopy() is called exclusively from the app thread.
//   ConsumeLatest() and MarkConsumed() are called exclusively from the runtime thread.
//   The handoff uses std::atomic<int> with release/acquire ordering.
class HoldingPen {
  public:
    static constexpr int kSlotCount = 3;

    struct SlotMetadata {
        XrTime   displayTime{0};
        XrPosef  renderPose{};  // pose at render time, used for Path B warp delta
    };

    struct ReadySlot {
        int          index;
        VkSemaphore  copyDone;   // binary; wait on GPU queue before reading image
        VkImage      image;
        SlotMetadata meta;
    };

    // appQueueFamily: queue family index used by the app thread for copy submits.
    // runtimeQueueFamily: queue family index used by the runtime thread for presentation.
    // If they differ, images are created with VK_SHARING_MODE_CONCURRENT.
    HoldingPen(VkPhysicalDevice physDevice,
               VkDevice device,
               VkQueue appQueue,
               uint32_t appQueueFamily,
               uint32_t runtimeQueueFamily,
               uint32_t width,
               uint32_t height,
               VkFormat colorFormat);

    ~HoldingPen();

    HoldingPen(const HoldingPen&) = delete;
    HoldingPen& operator=(const HoldingPen&) = delete;

    // Called from the app thread inside xrEndFrame.
    // Finds the oldest-consumed slot, submits a vkCmdCopyImage from appColorImage,
    // signals slot's copyDone semaphore on completion, updates m_latestReadySlot.
    // Returns immediately — no CPU wait.
    void SubmitCopy(VkImage appColorImage,
                    VkImageLayout appColorLayout,
                    XrTime displayTime,
                    XrPosef renderPose);

    // Called from the runtime thread.
    // Returns the latest ready slot if it is newer than the last consumed slot,
    // otherwise returns nullopt (deadline miss / no new frame).
    std::optional<ReadySlot> ConsumeLatest();

    // Called from the runtime thread after successfully submitting a slot.
    // Resets the slot's consumed fence so it can be reused by SubmitCopy.
    void MarkConsumed(int slotIndex);

    // Called during teardown. Drains the GPU queue then frees all Vulkan resources.
    void DrainAndDestroy();

    // Returns the warp output image (used by RuntimeThread for Path B output).
    VkImage WarpOutputImage() const { return m_warpOutputImage; }

  private:
    struct Slot {
        VkImage        image{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        VkImageView    view{VK_NULL_HANDLE};
        VkSemaphore    copyDone{VK_NULL_HANDLE};  // binary
        VkFence        consumed{VK_NULL_HANDLE};  // signaled by runtime thread
        SlotMetadata   meta{};
    };

    void AllocateImage(VkImage& outImage, VkDeviceMemory& outMemory,
                       VkImageUsageFlags usage,
                       const uint32_t* queueFamilies, uint32_t queueFamilyCount);
    void FreeResources() noexcept;

    VkPhysicalDevice m_physDevice{VK_NULL_HANDLE};
    VkDevice         m_device{VK_NULL_HANDLE};
    VkQueue          m_appQueue{VK_NULL_HANDLE};
    uint32_t         m_appQueueFamily{0};
    uint32_t         m_runtimeQueueFamily{0};
    uint32_t         m_width{0};
    uint32_t         m_height{0};
    VkFormat         m_format{VK_FORMAT_UNDEFINED};

    VkCommandPool                           m_copyPool{VK_NULL_HANDLE};
    std::array<VkCommandBuffer, kSlotCount> m_copyCmds{};
    std::array<Slot, kSlotCount>            m_slots{};

    // Warp output image (1 extra slot, written by PoseWarper in Path B).
    VkImage        m_warpOutputImage{VK_NULL_HANDLE};
    VkDeviceMemory m_warpOutputMemory{VK_NULL_HANDLE};

    // Set by app thread (release), read by runtime thread (acquire).
    std::atomic<int> m_latestReadySlot{-1};
    // Last slot index consumed by the runtime thread. Runtime-thread-only.
    int m_lastConsumedSlot{-1};
    // Next slot the app thread will write into. App-thread-only.
    int m_nextWriteSlot{0};
};

} // namespace openxr_api_layer
