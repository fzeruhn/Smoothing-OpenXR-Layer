#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

namespace interop {

// RAII wrapper for a Vulkan image whose memory is exported to CUDA.
//
// Construction allocates a VkImage with VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
// exports the Win32 handle, and imports it into CUDA as a CUarray.
//
// Prerequisites:
//   - CUDA must be initialized (cuInit called, context current) before constructing.
//   - VK_KHR_external_memory_win32 must be enabled on the VkDevice.
class SharedImage {
  public:
    SharedImage(VkDevice device, VkPhysicalDevice physDevice,
                uint32_t width, uint32_t height, VkFormat format);
    ~SharedImage();

    SharedImage(const SharedImage&) = delete;
    SharedImage& operator=(const SharedImage&) = delete;
    SharedImage(SharedImage&&) noexcept;
    SharedImage& operator=(SharedImage&&) noexcept;

    VkImage  vkImage()  const { return m_image; }
    CUarray  cuArray()  const { return m_cuArray; }
    uint32_t width()    const { return m_width; }
    uint32_t height()   const { return m_height; }

  private:
    VkDevice          m_device{VK_NULL_HANDLE};
    VkImage           m_image{VK_NULL_HANDLE};
    VkDeviceMemory    m_memory{VK_NULL_HANDLE};
    CUexternalMemory  m_extMem{nullptr};
    CUmipmappedArray  m_mipmappedArray{nullptr};
    CUarray           m_cuArray{nullptr};
    uint32_t          m_width{0};
    uint32_t          m_height{0};

    void destroy() noexcept;
};

// RAII wrapper for a Vulkan semaphore exported to CUDA.
//
// Allows CUDA to signal/wait on the same semaphore that Vulkan uses in
// VkSubmitInfo::pWaitSemaphores / pSignalSemaphores, enabling precise
// cross-API GPU synchronization without CPU round-trips.
//
// Prerequisites:
//   - VK_KHR_external_semaphore_win32 must be enabled on the VkDevice.
class SharedSemaphore {
  public:
    explicit SharedSemaphore(VkDevice device);
    ~SharedSemaphore();

    SharedSemaphore(const SharedSemaphore&) = delete;
    SharedSemaphore& operator=(const SharedSemaphore&) = delete;
    SharedSemaphore(SharedSemaphore&&) noexcept;
    SharedSemaphore& operator=(SharedSemaphore&&) noexcept;

    // Vulkan handle — use in VkSubmitInfo::pWaitSemaphores or pSignalSemaphores.
    VkSemaphore vkSemaphore() const { return m_semaphore; }

    // Enqueues a signal on `stream`. Vulkan can then wait on vkSemaphore().
    void signal(CUstream stream);

    // Enqueues a wait on `stream`. CUDA execution stalls until Vulkan signals.
    void wait(CUstream stream);

  private:
    VkDevice             m_device{VK_NULL_HANDLE};
    VkSemaphore          m_semaphore{VK_NULL_HANDLE};
    CUexternalSemaphore  m_extSem{nullptr};

    void destroy() noexcept;
};

} // namespace interop
