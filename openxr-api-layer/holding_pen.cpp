#include "pch.h"
#include "holding_pen.h"
#include <log.h>

#include <stdexcept>
#include <string>
#include <cstring>
#include <mutex>

namespace openxr_api_layer {

using namespace log;

// Defined in layer.cpp — protects the shared VkQueue used by both threads.
extern std::mutex g_queueMutex;

#define CHECK_VK(call)                                                             \
    do {                                                                           \
        VkResult _r = (call);                                                      \
        if (_r != VK_SUCCESS)                                                      \
            throw std::runtime_error(std::string("HoldingPen Vulkan error ") +    \
                                     std::to_string(_r) + " in " #call);          \
    } while (0)

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

HoldingPen::HoldingPen(VkPhysicalDevice physDevice,
                       VkDevice device,
                       VkQueue appQueue,
                       uint32_t appQueueFamily,
                       uint32_t runtimeQueueFamily,
                       uint32_t width,
                       uint32_t height,
                       VkFormat colorFormat)
    : m_physDevice(physDevice),
      m_device(device),
      m_appQueue(appQueue),
      m_appQueueFamily(appQueueFamily),
      m_runtimeQueueFamily(runtimeQueueFamily),
      m_width(width),
      m_height(height),
      m_format(colorFormat) {

    bool sharedFamily = (appQueueFamily == runtimeQueueFamily);
    uint32_t families[2] = {appQueueFamily, runtimeQueueFamily};
    uint32_t familyCount  = sharedFamily ? 1u : 2u;

    // Command pool for the app-thread copy submits.
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = appQueueFamily;
    poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK_VK(vkCreateCommandPool(device, &poolCI, nullptr, &m_copyPool));

    VkCommandBufferAllocateInfo cbAI{};
    cbAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAI.commandPool        = m_copyPool;
    cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = kSlotCount;
    CHECK_VK(vkAllocateCommandBuffers(device, &cbAI, m_copyCmds.data()));

    // Per-slot resources.
    for (int i = 0; i < kSlotCount; ++i) {
        Slot& s = m_slots[i];

        AllocateImage(s.image, s.memory,
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                      families, familyCount);

        VkImageViewCreateInfo viewCI{};
        viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCI.image                           = s.image;
        viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewCI.format                          = colorFormat;
        viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCI.subresourceRange.baseMipLevel   = 0;
        viewCI.subresourceRange.levelCount     = 1;
        viewCI.subresourceRange.baseArrayLayer = 0;
        viewCI.subresourceRange.layerCount     = 1;
        CHECK_VK(vkCreateImageView(device, &viewCI, nullptr, &s.view));

        // Binary semaphore (Vulkan default — no VkSemaphoreTypeCreateInfo needed).
        VkSemaphoreCreateInfo semCI{};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        CHECK_VK(vkCreateSemaphore(device, &semCI, nullptr, &s.copyDone));

        // Fence starts signaled so SubmitCopy can pick any slot on the first frame.
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        CHECK_VK(vkCreateFence(device, &fenceCI, nullptr, &s.consumed));
    }

    // Warp output image (Path B destination).
    AllocateImage(m_warpOutputImage, m_warpOutputMemory,
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_STORAGE_BIT |
                  VK_IMAGE_USAGE_SAMPLED_BIT,
                  families, familyCount);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

HoldingPen::~HoldingPen() {
    FreeResources();
}

// ---------------------------------------------------------------------------
// AllocateImage
// ---------------------------------------------------------------------------

void HoldingPen::AllocateImage(VkImage& outImage, VkDeviceMemory& outMemory,
                                VkImageUsageFlags usage,
                                const uint32_t* queueFamilies,
                                uint32_t queueFamilyCount) {
    VkImageCreateInfo imageCI{};
    imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType     = VK_IMAGE_TYPE_2D;
    imageCI.format        = m_format;
    imageCI.extent        = {m_width, m_height, 1};
    imageCI.mipLevels     = 1;
    imageCI.arrayLayers   = 1;
    imageCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage         = usage;
    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (queueFamilyCount > 1) {
        imageCI.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        imageCI.queueFamilyIndexCount = queueFamilyCount;
        imageCI.pQueueFamilyIndices   = queueFamilies;
    } else {
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    CHECK_VK(vkCreateImage(m_device, &imageCI, nullptr, &outImage));

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(m_device, outImage, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physDevice, &memProps);

    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    if (memTypeIndex == UINT32_MAX)
        throw std::runtime_error("HoldingPen: no device-local memory type available");

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    CHECK_VK(vkAllocateMemory(m_device, &allocInfo, nullptr, &outMemory));
    CHECK_VK(vkBindImageMemory(m_device, outImage, outMemory, 0));
}

// ---------------------------------------------------------------------------
// SubmitCopy (app thread)
// ---------------------------------------------------------------------------

bool HoldingPen::SubmitCopy(VkImage appColorImage,
                            VkImageLayout appColorLayout,
                            XrTime displayTime,
                            XrPosef renderPose) {
    // Find a consumed (available) slot. Rotate through the ring.
    int chosenSlot = -1;
    for (int attempt = 0; attempt < kSlotCount; ++attempt) {
        int candidate = m_nextWriteSlot % kSlotCount;
        m_nextWriteSlot = (m_nextWriteSlot + 1) % kSlotCount;
        if (vkGetFenceStatus(m_device, m_slots[candidate].consumed) == VK_SUCCESS) {
            chosenSlot = candidate;
            break;
        }
    }
    if (chosenSlot < 0) {
        // All slots in use — drop this frame rather than racing with an
        // in-flight GPU read on slot 0.
        Log("HoldingPen::SubmitCopy: all slots busy, dropping frame.\n");
        m_dropCount.fetch_add(1, std::memory_order_release);
        return false;
    }

    Slot& s = m_slots[chosenSlot];
    CHECK_VK(vkResetFences(m_device, 1, &s.consumed));

    VkCommandBuffer cmd = m_copyCmds[chosenSlot];
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkResetCommandBuffer(cmd, 0));
    CHECK_VK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Barrier: slot image UNDEFINED → TRANSFER_DST_OPTIMAL.
    VkImageMemoryBarrier toDst{};
    toDst.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toDst.srcAccessMask       = 0;
    toDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    toDst.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.image               = s.image;
    toDst.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toDst);

    // Barrier: app image → TRANSFER_SRC_OPTIMAL.
    VkImageMemoryBarrier toSrc{};
    toSrc.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toSrc.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toSrc.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    toSrc.oldLayout           = appColorLayout;
    toSrc.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSrc.image               = appColorImage;
    toSrc.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toSrc);

    // Copy.
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent         = {m_width, m_height, 1};
    vkCmdCopyImage(cmd,
        appColorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        s.image,       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region);

    // Barrier: slot image TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL.
    VkImageMemoryBarrier toRead{};
    toRead.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toRead.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    toRead.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    toRead.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toRead.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toRead.image               = s.image;
    toRead.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toRead);

    // Restore app image to its original layout.
    VkImageMemoryBarrier restoreApp{};
    restoreApp.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    restoreApp.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    restoreApp.dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    restoreApp.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    restoreApp.newLayout           = appColorLayout;
    restoreApp.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    restoreApp.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    restoreApp.image               = appColorImage;
    restoreApp.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, nullptr, 0, nullptr, 1, &restoreApp);

    CHECK_VK(vkEndCommandBuffer(cmd));

    // Submit — signal copyDone binary semaphore. No CPU wait.
    // Lock the shared queue mutex: the runtime thread may be submitting concurrently.
    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = &s.copyDone;
    {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        CHECK_VK(vkQueueSubmit(m_appQueue, 1, &submitInfo, VK_NULL_HANDLE));
    }

    const uint64_t frameId = m_nextFrameId++;
    s.meta = {frameId, displayTime, renderPose};
    m_latestSubmittedFrameId.store(frameId, std::memory_order_release);
    m_latestReadySlot.store(chosenSlot, std::memory_order_release);
    return true;
}

// ---------------------------------------------------------------------------
// ConsumeLatest (runtime thread)
// ---------------------------------------------------------------------------

std::optional<HoldingPen::ReadySlot> HoldingPen::ConsumeLatest() {
    int latest = m_latestReadySlot.load(std::memory_order_acquire);
    if (latest < 0 || latest == m_lastConsumedSlot) {
        return std::nullopt;
    }
    m_lastConsumedSlot = latest;
    const Slot& s = m_slots[latest];
    m_lastConsumedFrameId.store(s.meta.frameId, std::memory_order_release);
    return ReadySlot{latest, s.copyDone, s.image, s.meta};
}

// ---------------------------------------------------------------------------
// DrainAndDestroy
// ---------------------------------------------------------------------------

void HoldingPen::DrainAndDestroy() {
    if (m_appQueue != VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> lock(g_queueMutex);
        vkQueueWaitIdle(m_appQueue);
    }
    FreeResources();
}

// ---------------------------------------------------------------------------
// FreeResources
// ---------------------------------------------------------------------------

void HoldingPen::FreeResources() noexcept {
    if (m_device == VK_NULL_HANDLE) return;

    for (Slot& s : m_slots) {
        if (s.consumed)  { vkDestroyFence(m_device, s.consumed, nullptr);       s.consumed  = VK_NULL_HANDLE; }
        if (s.copyDone)  { vkDestroySemaphore(m_device, s.copyDone, nullptr);   s.copyDone  = VK_NULL_HANDLE; }
        if (s.view)      { vkDestroyImageView(m_device, s.view, nullptr);        s.view      = VK_NULL_HANDLE; }
        if (s.image)     { vkDestroyImage(m_device, s.image, nullptr);           s.image     = VK_NULL_HANDLE; }
        if (s.memory)    { vkFreeMemory(m_device, s.memory, nullptr);            s.memory    = VK_NULL_HANDLE; }
    }

    if (m_warpOutputImage)  { vkDestroyImage(m_device, m_warpOutputImage, nullptr);  m_warpOutputImage  = VK_NULL_HANDLE; }
    if (m_warpOutputMemory) { vkFreeMemory(m_device, m_warpOutputMemory, nullptr);   m_warpOutputMemory = VK_NULL_HANDLE; }

    if (m_copyPool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_device, m_copyPool, kSlotCount, m_copyCmds.data());
        vkDestroyCommandPool(m_device, m_copyPool, nullptr);
        m_copyPool = VK_NULL_HANDLE;
    }

    m_device = VK_NULL_HANDLE;
}

} // namespace openxr_api_layer
