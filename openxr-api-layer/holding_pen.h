#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <array>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <string>

namespace openxr_api_layer {

// Three-slot ring buffer of layer-owned VkImages.
// App thread: calls AcquireIdleSlot() + SubmitCopy() to deposit a frame.
// Runtime thread: calls PollSlots() + GetFreshestReadySlot() + ReleaseSlot() to consume a frame.
// Thread safety: slot state transitions are atomic. vkQueueSubmit is serialized externally.
class HoldingPen {
  public:
    static constexpr uint32_t kSlotCount = 3;

    enum class SlotState : uint32_t {
        Idle         = 0,  // Available for app thread
        CopyPending  = 1,  // App submitted copy, fence not yet signaled
        Ready        = 2,  // Fence signaled, available for runtime thread
        RuntimeUse   = 3,  // Runtime thread currently submitting this slot
    };

    struct Slot {
        VkImage         image{VK_NULL_HANDLE};
        VkDeviceMemory  memory{VK_NULL_HANDLE};
        VkFence         copyDoneFence{VK_NULL_HANDLE};
        VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
        XrTime          displayTime{0};
        uint64_t        frameSeq{0};  // monotonic counter for "freshest"
        std::atomic<SlotState> state{SlotState::Idle};
    };

    bool Initialize(VkDevice device, VkPhysicalDevice physDevice,
                    uint32_t width, uint32_t height, VkFormat format,
                    uint32_t queueFamilyIndex);
    void Shutdown();

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }

    // App thread: returns slot index [0,kSlotCount) or -1 if all busy.
    int32_t AcquireIdleSlot();

    // App thread: records and submits vkCmdCopyImage for the slot.
    // queue must be externally serialized via queueMutex.
    // sourceColor must be in VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL by caller.
    // After return, the slot is in CopyPending state.
    // Returns false on Vulkan error.
    bool SubmitCopy(int32_t slotIndex,
                    VkImage sourceColor,
                    uint32_t width, uint32_t height,
                    XrTime displayTime,
                    VkQueue queue, std::mutex& queueMutex);

    // Either thread: poll VkFence for all CopyPending slots; transitions them to Ready on completion.
    void PollSlots();

    // Runtime thread: returns index of Ready slot with highest frameSeq, or -1.
    // Stale Ready slots are returned to Idle.
    int32_t GetFreshestReadySlot();

    // Runtime thread: transitions slot back to Idle.
    void ReleaseSlot(int32_t slotIndex);

    VkImage  GetSlotImage(int32_t slotIndex) const;
    XrTime   GetSlotDisplayTime(int32_t slotIndex) const;

    std::string GetDebugState() const;  // for logging

  private:
    bool AllocateSlot(uint32_t slotIndex, VkPhysicalDevice physDevice,
                      VkFormat format, uint32_t width, uint32_t height);

    VkDevice      m_device{VK_NULL_HANDLE};
    VkCommandPool m_commandPool{VK_NULL_HANDLE};
    VkFormat      m_format{VK_FORMAT_UNDEFINED};
    uint32_t      m_width{0};
    uint32_t      m_height{0};
    bool          m_initialized{false};
    uint64_t      m_frameSeqCounter{0};
    std::array<Slot, kSlotCount> m_slots{};
};

} // namespace openxr_api_layer
