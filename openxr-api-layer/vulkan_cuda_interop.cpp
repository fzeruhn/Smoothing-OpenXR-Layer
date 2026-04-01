#define VK_USE_PLATFORM_WIN32_KHR
#include "vulkan_cuda_interop.h"

#include <vulkan/vulkan_win32.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define CHECK_VK(call)                                                          \
    do {                                                                        \
        VkResult _r = (call);                                                   \
        if (_r != VK_SUCCESS)                                                   \
            throw std::runtime_error(std::string("Vulkan error ") +             \
                                     std::to_string(_r) + " in " #call);       \
    } while (0)

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        CUresult _r = (call);                                                   \
        if (_r != CUDA_SUCCESS) {                                               \
            const char* str = nullptr;                                          \
            cuGetErrorString(_r, &str);                                         \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     (str ? str : "unknown") + " in " #call);  \
        }                                                                       \
    } while (0)

namespace interop {

// ---------------------------------------------------------------------------
// SharedImage
// ---------------------------------------------------------------------------

SharedImage::SharedImage(VkDevice device, VkPhysicalDevice physDevice,
                         uint32_t width, uint32_t height, VkFormat format)
    : m_device(device), m_width(width), m_height(height) {

    // Extension functions are not linked directly — load at runtime.
    auto fnGetMemoryHandle =
        reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    if (!fnGetMemoryHandle)
        throw std::runtime_error(
            "vkGetMemoryWin32HandleKHR not available — "
            "VK_KHR_external_memory_win32 must be enabled at VkDevice creation");

    // 1. Create VkImage with external memory export flag.
    VkExternalMemoryImageCreateInfo extImageInfo{};
    extImageInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    extImageInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext         = &extImageInfo;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = format;
    imageInfo.extent        = {width, height, 1};
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_STORAGE_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    CHECK_VK(vkCreateImage(device, &imageInfo, nullptr, &m_image));

    // 2. Find a device-local memory type satisfying image requirements.
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, m_image, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);

    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    if (memTypeIndex == UINT32_MAX)
        throw std::runtime_error(
            "No device-local memory type satisfies image requirements");

    // 3. Allocate memory with Win32 export flag.
    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext           = &exportInfo;
    allocInfo.allocationSize  = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, nullptr, &m_memory));
    CHECK_VK(vkBindImageMemory(device, m_image, m_memory, 0));

    // 4. Export the Win32 handle.
    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory     = m_memory;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE win32Handle    = nullptr;
    CHECK_VK(fnGetMemoryHandle(device, &handleInfo, &win32Handle));

    // 5. Import into CUDA. CloseHandle after import — CUDA takes ownership.
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC extMemDesc{};
    extMemDesc.type                = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    extMemDesc.handle.win32.handle = win32Handle;
    extMemDesc.size                = memReqs.size;
    CUresult cuResult = cuImportExternalMemory(&m_extMem, &extMemDesc);
    CloseHandle(win32Handle);
    CHECK_CUDA(cuResult);

    // 6. Map to a CUarray (mip level 0).
    //    CUDA_ARRAY3D_SURFACE_LDST is required for surf2Dwrite in kernels.
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC arrayDesc{};
    arrayDesc.offset                = 0;
    arrayDesc.arrayDesc.Width       = width;
    arrayDesc.arrayDesc.Height      = height;
    arrayDesc.arrayDesc.Depth       = 0;
    arrayDesc.arrayDesc.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
    arrayDesc.arrayDesc.NumChannels = 4;  // RGBA
    arrayDesc.arrayDesc.Flags       = CUDA_ARRAY3D_SURFACE_LDST;
    arrayDesc.numLevels             = 1;
    CHECK_CUDA(cuExternalMemoryGetMappedMipmappedArray(
        &m_mipmappedArray, m_extMem, &arrayDesc));
    CHECK_CUDA(cuMipmappedArrayGetLevel(&m_cuArray, m_mipmappedArray, 0));
}

SharedImage::~SharedImage() {
    destroy();
}

void SharedImage::destroy() noexcept {
    // Release CUDA resources before Vulkan — CUDA must stop referencing the
    // memory before Vulkan frees the underlying allocation.
    m_cuArray = nullptr;  // owned by m_mipmappedArray, not separately destroyed
    if (m_mipmappedArray) { cuMipmappedArrayDestroy(m_mipmappedArray); m_mipmappedArray = nullptr; }
    if (m_extMem)         { cuDestroyExternalMemory(m_extMem);          m_extMem = nullptr; }
    if (m_memory)         { vkFreeMemory(m_device, m_memory, nullptr);  m_memory = VK_NULL_HANDLE; }
    if (m_image)          { vkDestroyImage(m_device, m_image, nullptr); m_image  = VK_NULL_HANDLE; }
}

SharedImage::SharedImage(SharedImage&& other) noexcept
    : m_device(other.m_device),
      m_image(other.m_image),
      m_memory(other.m_memory),
      m_extMem(other.m_extMem),
      m_mipmappedArray(other.m_mipmappedArray),
      m_cuArray(other.m_cuArray),
      m_width(other.m_width),
      m_height(other.m_height) {
    other.m_device        = VK_NULL_HANDLE;
    other.m_image         = VK_NULL_HANDLE;
    other.m_memory        = VK_NULL_HANDLE;
    other.m_extMem        = nullptr;
    other.m_mipmappedArray = nullptr;
    other.m_cuArray       = nullptr;
}

SharedImage& SharedImage::operator=(SharedImage&& other) noexcept {
    if (this != &other) {
        destroy();
        m_device         = other.m_device;
        m_image          = other.m_image;
        m_memory         = other.m_memory;
        m_extMem         = other.m_extMem;
        m_mipmappedArray = other.m_mipmappedArray;
        m_cuArray        = other.m_cuArray;
        m_width          = other.m_width;
        m_height         = other.m_height;
        other.m_device        = VK_NULL_HANDLE;
        other.m_image         = VK_NULL_HANDLE;
        other.m_memory        = VK_NULL_HANDLE;
        other.m_extMem        = nullptr;
        other.m_mipmappedArray = nullptr;
        other.m_cuArray       = nullptr;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// SharedSemaphore
// ---------------------------------------------------------------------------

SharedSemaphore::SharedSemaphore(VkDevice device) : m_device(device) {
    auto fnGetSemaphoreHandle =
        reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
    if (!fnGetSemaphoreHandle)
        throw std::runtime_error(
            "vkGetSemaphoreWin32HandleKHR not available — "
            "VK_KHR_external_semaphore_win32 must be enabled at VkDevice creation");

    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semInfo.pNext = &exportInfo;
    CHECK_VK(vkCreateSemaphore(device, &semInfo, nullptr, &m_semaphore));

    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore  = m_semaphore;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE win32Handle = nullptr;
    CHECK_VK(fnGetSemaphoreHandle(device, &handleInfo, &win32Handle));

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC extSemDesc{};
    extSemDesc.type                = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
    extSemDesc.handle.win32.handle = win32Handle;
    CUresult cuResult = cuImportExternalSemaphore(&m_extSem, &extSemDesc);
    CloseHandle(win32Handle);
    CHECK_CUDA(cuResult);
}

SharedSemaphore::~SharedSemaphore() {
    destroy();
}

void SharedSemaphore::destroy() noexcept {
    if (m_extSem)    { cuDestroyExternalSemaphore(m_extSem);              m_extSem    = nullptr; }
    if (m_semaphore) { vkDestroySemaphore(m_device, m_semaphore, nullptr); m_semaphore = VK_NULL_HANDLE; }
}

SharedSemaphore::SharedSemaphore(SharedSemaphore&& other) noexcept
    : m_device(other.m_device),
      m_semaphore(other.m_semaphore),
      m_extSem(other.m_extSem) {
    other.m_device    = VK_NULL_HANDLE;
    other.m_semaphore = VK_NULL_HANDLE;
    other.m_extSem    = nullptr;
}

SharedSemaphore& SharedSemaphore::operator=(SharedSemaphore&& other) noexcept {
    if (this != &other) {
        destroy();
        m_device    = other.m_device;
        m_semaphore = other.m_semaphore;
        m_extSem    = other.m_extSem;
        other.m_device    = VK_NULL_HANDLE;
        other.m_semaphore = VK_NULL_HANDLE;
        other.m_extSem    = nullptr;
    }
    return *this;
}

void SharedSemaphore::signal(CUstream stream) {
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
    CHECK_CUDA(cuSignalExternalSemaphoresAsync(&m_extSem, &params, 1, stream));
}

void SharedSemaphore::wait(CUstream stream) {
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
    CHECK_CUDA(cuWaitExternalSemaphoresAsync(&m_extSem, &params, 1, stream));
}

} // namespace interop
