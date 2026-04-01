#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "vulkan_cuda_interop.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declaration of the kernel launcher (compiled in fill_pattern.cu).
// The <<<>>> launch syntax is only valid in .cu files compiled by nvcc.
// ---------------------------------------------------------------------------
extern "C" void launch_fill_pattern(CUsurfObject surf, int width, int height);

// ---------------------------------------------------------------------------
// Minimal error-checking helpers
// ---------------------------------------------------------------------------

static void check_vk(VkResult r, const char* where) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[FAIL] Vulkan error %d at %s\n", r, where);
        exit(1);
    }
}
#define VK(call) check_vk((call), #call)

static void check_cu(CUresult r, const char* where) {
    if (r != CUDA_SUCCESS) {
        const char* str = nullptr;
        cuGetErrorString(r, &str);
        fprintf(stderr, "[FAIL] CUDA error %s at %s\n", str ? str : "?", where);
        exit(1);
    }
}
#define CU(call) check_cu((call), #call)

// ---------------------------------------------------------------------------
// Headless Vulkan context
// ---------------------------------------------------------------------------

struct VulkanContext {
    VkInstance       instance{VK_NULL_HANDLE};
    VkPhysicalDevice physDevice{VK_NULL_HANDLE};
    VkDevice         device{VK_NULL_HANDLE};
    VkQueue          queue{VK_NULL_HANDLE};
    uint32_t         queueFamily{0};
    VkCommandPool    cmdPool{VK_NULL_HANDLE};
    VkCommandBuffer  cmdBuf{VK_NULL_HANDLE};
};

static VulkanContext createVulkan() {
    VulkanContext ctx;

    // Instance — request external memory/semaphore capability query extensions.
    const char* instExts[] = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };
    VkApplicationInfo appInfo{};
    appInfo.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo        = &appInfo;
    instInfo.enabledExtensionCount   = 3;
    instInfo.ppEnabledExtensionNames = instExts;
    VK(vkCreateInstance(&instInfo, nullptr, &ctx.instance));

    // Physical device — prefer discrete GPU.
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, nullptr);
    if (devCount == 0) throw std::runtime_error("No Vulkan-capable GPU found");
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, devs.data());

    for (auto pd : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx.physDevice = pd;
            printf("Using GPU: %s\n", props.deviceName);
            break;
        }
    }
    if (!ctx.physDevice) {
        ctx.physDevice = devs[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(ctx.physDevice, &props);
        printf("Using GPU (fallback): %s\n", props.deviceName);
    }

    // Queue family — any family supporting compute + transfer.
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physDevice, &qfCount, qfs.data());

    ctx.queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (qfs[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) {
            ctx.queueFamily = i;
            break;
        }
    }
    if (ctx.queueFamily == UINT32_MAX)
        throw std::runtime_error("No compute/transfer queue family found");

    // Device — enable external memory and semaphore Win32 extensions.
    const char* devExts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
    };
    float qPriority = 1.0f;
    VkDeviceQueueCreateInfo qInfo{};
    qInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = ctx.queueFamily;
    qInfo.queueCount       = 1;
    qInfo.pQueuePriorities = &qPriority;

    VkDeviceCreateInfo devInfo{};
    devInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount    = 1;
    devInfo.pQueueCreateInfos       = &qInfo;
    devInfo.enabledExtensionCount   = 4;
    devInfo.ppEnabledExtensionNames = devExts;
    VK(vkCreateDevice(ctx.physDevice, &devInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);

    // Command pool + buffer for the readback step.
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = ctx.queueFamily;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.cmdPool));

    VkCommandBufferAllocateInfo cbInfo{};
    cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool        = ctx.cmdPool;
    cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    VK(vkAllocateCommandBuffers(ctx.device, &cbInfo, &ctx.cmdBuf));

    return ctx;
}

static void destroyVulkan(VulkanContext& ctx) {
    if (ctx.cmdPool)  vkDestroyCommandPool(ctx.device, ctx.cmdPool, nullptr);
    if (ctx.device)   vkDestroyDevice(ctx.device, nullptr);
    if (ctx.instance) vkDestroyInstance(ctx.instance, nullptr);
}

static uint32_t findHostVisibleMemType(VkPhysicalDevice pd, uint32_t typeBits) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            return i;
        }
    }
    throw std::runtime_error("No host-visible coherent memory type found");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    const int W = 256, H = 256;

    try {
        // 1. Initialise CUDA driver API.
        // Use the primary context — cuCtxCreate signature changed in CUDA 13.x.
        CU(cuInit(0));
        CUdevice cuDev;
        CU(cuDeviceGet(&cuDev, 0));
        CUcontext cuCtx;
        CU(cuDevicePrimaryCtxRetain(&cuCtx, cuDev));
        CU(cuCtxSetCurrent(cuCtx));

        // 2. Create headless Vulkan context.
        VulkanContext vk = createVulkan();

        bool pass = false;
        // Nested scope so image/sem/readback destructors fire before destroyVulkan.
        {
            // 3. Construct shared resources (throws on any setup failure).
            interop::SharedImage     image(vk.device, vk.physDevice, W, H,
                                           VK_FORMAT_R8G8B8A8_UNORM);
            interop::SharedSemaphore sem(vk.device);

            // 4. Create a CUDA surface object wrapping the shared CUarray.
            CUDA_RESOURCE_DESC resDesc{};
            resDesc.resType          = CU_RESOURCE_TYPE_ARRAY;
            resDesc.res.array.hArray = image.cuArray();
            CUsurfObject surf = 0;
            CU(cuSurfObjectCreate(&surf, &resDesc));

            // 5. Launch the fill kernel on the default stream.
            launch_fill_pattern(surf, W, H);

            // 6. Signal the shared semaphore from CUDA so Vulkan can wait on it.
            //    This enqueues the signal after the kernel on stream 0.
            sem.signal(0);

            CU(cuSurfObjectDestroy(surf));

            // 7. Allocate a host-visible readback buffer.
            VkDeviceSize bufSize = static_cast<VkDeviceSize>(W) * H * 4;

            VkBufferCreateInfo bufInfo{};
            bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufInfo.size        = bufSize;
            bufInfo.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer readbackBuf;
            VK(vkCreateBuffer(vk.device, &bufInfo, nullptr, &readbackBuf));

            VkMemoryRequirements bufMemReqs;
            vkGetBufferMemoryRequirements(vk.device, readbackBuf, &bufMemReqs);
            uint32_t memIdx = findHostVisibleMemType(vk.physDevice,
                                                     bufMemReqs.memoryTypeBits);
            VkMemoryAllocateInfo bufAlloc{};
            bufAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            bufAlloc.allocationSize  = bufMemReqs.size;
            bufAlloc.memoryTypeIndex = memIdx;
            VkDeviceMemory readbackMem;
            VK(vkAllocateMemory(vk.device, &bufAlloc, nullptr, &readbackMem));
            VK(vkBindBufferMemory(vk.device, readbackBuf, readbackMem, 0));

            // 8. Record: wait on CUDA semaphore, transition image, copy to buffer.
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK(vkBeginCommandBuffer(vk.cmdBuf, &beginInfo));

            // Transition UNDEFINED → TRANSFER_SRC_OPTIMAL.
            VkImageMemoryBarrier barrier{};
            barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask       = 0;
            barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image               = image.vkImage();
            barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(vk.cmdBuf,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);

            VkBufferImageCopy region{};
            region.bufferOffset      = 0;
            region.bufferRowLength   = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageOffset       = {0, 0, 0};
            region.imageExtent       = {static_cast<uint32_t>(W),
                                        static_cast<uint32_t>(H), 1};
            vkCmdCopyImageToBuffer(vk.cmdBuf,
                                   image.vkImage(),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   readbackBuf, 1, &region);

            VK(vkEndCommandBuffer(vk.cmdBuf));

            // Submit: Vulkan waits for the CUDA semaphore signal before the copy runs.
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSemaphore          waitSem   = sem.vkSemaphore();
            VkSubmitInfo submitInfo{};
            submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores    = &waitSem;
            submitInfo.pWaitDstStageMask  = &waitStage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers    = &vk.cmdBuf;
            VK(vkQueueSubmit(vk.queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK(vkQueueWaitIdle(vk.queue));

            // 9. Map and verify every pixel.
            void* mapped = nullptr;
            VK(vkMapMemory(vk.device, readbackMem, 0, bufSize, 0, &mapped));
            auto* pixels = reinterpret_cast<unsigned char*>(mapped);

            pass = true;
            for (int y = 0; y < H && pass; ++y) {
                for (int x = 0; x < W && pass; ++x) {
                    int     idx  = (y * W + x) * 4;
                    uint8_t expR = static_cast<uint8_t>((x + y * W) % 256);
                    uint8_t gotR = pixels[idx];
                    if (gotR != expR) {
                        printf("[FAIL] Pixel (%d,%d): expected R=%u, got R=%u\n",
                               x, y, expR, gotR);
                        pass = false;
                    }
                }
            }
            vkUnmapMemory(vk.device, readbackMem);

            // 10. Cleanup readback resources (image/sem destroy when scope closes).
            vkFreeMemory(vk.device, readbackMem, nullptr);
            vkDestroyBuffer(vk.device, readbackBuf, nullptr);
        } // image and sem destructors fire here, before destroyVulkan

        destroyVulkan(vk);
        cuDevicePrimaryCtxRelease(cuDev);

        if (pass) {
            printf("[PASS] Vulkan/CUDA interop verified (%dx%d RGBA, "
                   "pattern round-trip)\n", W, H);
            return 0;
        }
        return 1;

    } catch (const std::exception& e) {
        fprintf(stderr, "[FAIL] Exception: %s\n", e.what());
        return 1;
    }
}
