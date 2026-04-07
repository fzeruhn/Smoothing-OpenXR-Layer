#include "pch.h"
#include "holding_pen.h"
#include <log.h>
#include <fmt/format.h>

namespace openxr_api_layer {

using namespace log;

bool HoldingPen::Initialize(VkDevice device, VkPhysicalDevice physDevice,
                             uint32_t width, uint32_t height, VkFormat format,
                             uint32_t queueFamilyIndex) {
    if (m_initialized) return true;
    if (!device || !physDevice || width == 0 || height == 0) return false;

    m_device = device;
    m_width  = width;
    m_height = height;
    m_format = format;

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        Log("[HoldingPen] Failed to create command pool\n");
        return false;
    }

    VkCommandBufferAllocateInfo cbInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbInfo.commandPool        = m_commandPool;
    cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = kSlotCount;
    std::array<VkCommandBuffer, kSlotCount> cbs{};
    if (vkAllocateCommandBuffers(device, &cbInfo, cbs.data()) != VK_SUCCESS) {
        Log("[HoldingPen] Failed to allocate command buffers\n");
        vkDestroyCommandPool(device, m_commandPool, nullptr);
        m_commandPool = VK_NULL_HANDLE;
        return false;
    }

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < kSlotCount; ++i) {
        m_slots[i].commandBuffer = cbs[i];
        m_slots[i].state.store(SlotState::Idle);
        m_slots[i].frameSeq    = 0;
        m_slots[i].displayTime = 0;

        if (vkCreateFence(device, &fenceInfo, nullptr, &m_slots[i].copyDoneFence) != VK_SUCCESS) {
            Log(fmt::format("[HoldingPen] Failed to create fence for slot {}\n", i));
            Shutdown();
            return false;
        }

        if (!AllocateSlot(i, physDevice, format, width, height)) {
            Log(fmt::format("[HoldingPen] Failed to allocate image for slot {}\n", i));
            Shutdown();
            return false;
        }
    }

    m_initialized = true;
    Log(fmt::format("[HoldingPen] Initialized: {}x{} fmt={} slots={}\n",
                    width, height, static_cast<int>(format), kSlotCount));
    return true;
}

bool HoldingPen::AllocateSlot(uint32_t i, VkPhysicalDevice physDevice,
                               VkFormat format, uint32_t width, uint32_t height) {
    VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgInfo.imageType   = VK_IMAGE_TYPE_2D;
    imgInfo.format      = format;
    imgInfo.extent      = {width, height, 1};
    imgInfo.mipLevels   = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage       = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    imgInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(m_device, &imgInfo, nullptr, &m_slots[i].image) != VK_SUCCESS) return false;

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(m_device, m_slots[i].image, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);

    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t j = 0; j < memProps.memoryTypeCount; ++j) {
        if ((memReqs.memoryTypeBits & (1u << j)) &&
            (memProps.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = j;
            break;
        }
    }
    if (memTypeIndex == UINT32_MAX) {
        vkDestroyImage(m_device, m_slots[i].image, nullptr);
        m_slots[i].image = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize  = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &m_slots[i].memory) != VK_SUCCESS) {
        vkDestroyImage(m_device, m_slots[i].image, nullptr);
        m_slots[i].image = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindImageMemory(m_device, m_slots[i].image, m_slots[i].memory, 0) != VK_SUCCESS) {
        vkFreeMemory(m_device, m_slots[i].memory, nullptr);
        vkDestroyImage(m_device, m_slots[i].image, nullptr);
        m_slots[i].image  = VK_NULL_HANDLE;
        m_slots[i].memory = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

void HoldingPen::Shutdown() {
    if (!m_device) return;
    vkDeviceWaitIdle(m_device);

    for (auto& slot : m_slots) {
        if (slot.copyDoneFence) {
            vkDestroyFence(m_device, slot.copyDoneFence, nullptr);
            slot.copyDoneFence = VK_NULL_HANDLE;
        }
        if (slot.memory) {
            vkFreeMemory(m_device, slot.memory, nullptr);
            slot.memory = VK_NULL_HANDLE;
        }
        if (slot.image) {
            vkDestroyImage(m_device, slot.image, nullptr);
            slot.image = VK_NULL_HANDLE;
        }
        slot.state.store(SlotState::Idle);
    }
    if (m_commandPool) {
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        m_commandPool = VK_NULL_HANDLE;
    }
    m_initialized = false;
}

int32_t HoldingPen::AcquireIdleSlot() {
    for (uint32_t i = 0; i < kSlotCount; ++i) {
        SlotState expected = SlotState::Idle;
        if (m_slots[i].state.compare_exchange_strong(expected, SlotState::CopyPending)) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

bool HoldingPen::SubmitCopy(int32_t slotIndex,
                              VkImage sourceColor,
                              uint32_t width, uint32_t height,
                              XrTime displayTime,
                              VkQueue queue, std::mutex& queueMutex) {
    if (slotIndex < 0 || slotIndex >= static_cast<int32_t>(kSlotCount)) return false;
    auto& slot = m_slots[slotIndex];

    // Reset fence before recording (fence starts signaled from VK_FENCE_CREATE_SIGNALED_BIT).
    if (vkResetFences(m_device, 1, &slot.copyDoneFence) != VK_SUCCESS) {
        slot.state.store(SlotState::Idle);
        return false;
    }

    VkCommandBuffer cmd = slot.commandBuffer;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) {
        slot.state.store(SlotState::Idle);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
        slot.state.store(SlotState::Idle);
        return false;
    }

    // Transition source COLOR_ATTACHMENT_OPTIMAL → TRANSFER_SRC_OPTIMAL
    VkImageMemoryBarrier srcBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    srcBarrier.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    srcBarrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    srcBarrier.oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    srcBarrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    srcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    srcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    srcBarrier.image               = sourceColor;
    srcBarrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // Transition holding-pen slot UNDEFINED → TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier dstBarrier      = srcBarrier;
    dstBarrier.srcAccessMask = 0;
    dstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dstBarrier.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    dstBarrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dstBarrier.image         = slot.image;

    VkImageMemoryBarrier toTransfer[2] = {srcBarrier, dstBarrier};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 2, toTransfer);

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent         = {width, height, 1};
    vkCmdCopyImage(cmd,
        sourceColor, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        slot.image,  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region);

    // Restore source TRANSFER_SRC_OPTIMAL → COLOR_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier restoreSrc      = srcBarrier;
    restoreSrc.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    restoreSrc.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    restoreSrc.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    restoreSrc.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Transition holding-pen slot TRANSFER_DST_OPTIMAL → COLOR_ATTACHMENT_OPTIMAL (ready for compositor)
    VkImageMemoryBarrier readyDst      = dstBarrier;
    readyDst.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    readyDst.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    readyDst.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    readyDst.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkImageMemoryBarrier toFinal[2] = {restoreSrc, readyDst};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, nullptr, 0, nullptr, 2, toFinal);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        slot.state.store(SlotState::Idle);
        return false;
    }

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (vkQueueSubmit(queue, 1, &submitInfo, slot.copyDoneFence) != VK_SUCCESS) {
            slot.state.store(SlotState::Idle);
            return false;
        }
    }

    slot.displayTime = displayTime;
    slot.frameSeq    = ++m_frameSeqCounter;
    Log(fmt::format("[HoldingPen] Slot {} copy submitted seq={} src={:p} dst={:p}\n",
                    slotIndex, slot.frameSeq,
                    static_cast<void*>(sourceColor),
                    static_cast<void*>(slot.image)));
    return true;
}

void HoldingPen::PollSlots() {
    for (uint32_t i = 0; i < kSlotCount; ++i) {
        if (m_slots[i].state.load() != SlotState::CopyPending) continue;
        const VkResult status = vkGetFenceStatus(m_device, m_slots[i].copyDoneFence);
        if (status == VK_SUCCESS) {
            m_slots[i].state.store(SlotState::Ready);
            Log(fmt::format("[HoldingPen] Slot {} fence signaled → Ready seq={}\n",
                            i, m_slots[i].frameSeq));
        }
    }
}

int32_t HoldingPen::GetFreshestReadySlot() {
    int32_t  best    = -1;
    uint64_t bestSeq = 0;
    for (uint32_t i = 0; i < kSlotCount; ++i) {
        if (m_slots[i].state.load() == SlotState::Ready && m_slots[i].frameSeq > bestSeq) {
            best    = static_cast<int32_t>(i);
            bestSeq = m_slots[i].frameSeq;
        }
    }
    if (best >= 0) {
        SlotState expected = SlotState::Ready;
        m_slots[best].state.compare_exchange_strong(expected, SlotState::RuntimeUse);
    }
    // Release any older Ready slots back to Idle (stale frames)
    for (uint32_t i = 0; i < kSlotCount; ++i) {
        if (static_cast<int32_t>(i) != best) {
            SlotState expected = SlotState::Ready;
            m_slots[i].state.compare_exchange_strong(expected, SlotState::Idle);
        }
    }
    return best;
}

void HoldingPen::ReleaseSlot(int32_t slotIndex) {
    if (slotIndex < 0 || slotIndex >= static_cast<int32_t>(kSlotCount)) return;
    m_slots[slotIndex].state.store(SlotState::Idle);
    Log(fmt::format("[HoldingPen] Slot {} released → Idle\n", slotIndex));
}

VkImage HoldingPen::GetSlotImage(int32_t slotIndex) const {
    if (slotIndex < 0 || slotIndex >= static_cast<int32_t>(kSlotCount)) return VK_NULL_HANDLE;
    return m_slots[slotIndex].image;
}

XrTime HoldingPen::GetSlotDisplayTime(int32_t slotIndex) const {
    if (slotIndex < 0 || slotIndex >= static_cast<int32_t>(kSlotCount)) return 0;
    return m_slots[slotIndex].displayTime;
}

std::string HoldingPen::GetDebugState() const {
    std::string s;
    for (uint32_t i = 0; i < kSlotCount; ++i) {
        const char* stateName = "?";
        switch (m_slots[i].state.load()) {
            case SlotState::Idle:        stateName = "Idle"; break;
            case SlotState::CopyPending: stateName = "Pend"; break;
            case SlotState::Ready:       stateName = "Rdy";  break;
            case SlotState::RuntimeUse:  stateName = "RT";   break;
        }
        s += fmt::format("[{}:{}:seq{}]", i, stateName, m_slots[i].frameSeq);
    }
    return s;
}

} // namespace openxr_api_layer
