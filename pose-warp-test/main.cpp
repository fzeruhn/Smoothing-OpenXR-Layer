// pose-warp-test/main.cpp
// Validates PoseWarper with synthetic input:
//   Input: 256x256 RGBA checkerboard pattern
//   Transform: 5° yaw rotation (rotation around Y axis)
//   Expected output: Rotated checkerboard with bilinear interpolation
//
// Pass criteria: At least 20% of central ROI pixels differ from input

#include "pose_warp.h"
#include "pose_warp_math.h"
#include "vulkan_cuda_interop.h"

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iterator>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

static constexpr int W = 256;
static constexpr int H = 256;
static constexpr float YAW_DEG = 5.0f;  // 5° yaw rotation
static constexpr float YAW_RAD = YAW_DEG * 3.14159265f / 180.0f;

// Central 80% ROI used for change detection
static constexpr int MARGIN = W / 10;  // 10% margin = 25 pixels
static constexpr int ROI_X0 = MARGIN;
static constexpr int ROI_Y0 = MARGIN;
static constexpr int ROI_X1 = W - MARGIN;
static constexpr int ROI_Y1 = H - MARGIN;
static constexpr float MIN_CHANGED_PERCENT = 20.0f;

// Helper: Create quaternion for yaw rotation (rotation around Y axis)
static XrQuaternionf makeYawRotation(float yaw_rad) {
    XrQuaternionf q;
    q.x = 0.0f;
    q.y = std::sin(yaw_rad / 2.0f);
    q.z = 0.0f;
    q.w = std::cos(yaw_rad / 2.0f);
    return q;
}

// Helper: Generate checkerboard pattern (8x8 pixel blocks, alternating white/black)
static void generateCheckerboard(std::vector<uint8_t>& rgba, int width, int height) {
    rgba.resize(width * height * 4);
    constexpr int BLOCK_SIZE = 8;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int block_x = x / BLOCK_SIZE;
            int block_y = y / BLOCK_SIZE;
            bool isWhite = ((block_x + block_y) % 2 == 0);
            
            uint8_t val = isWhite ? 255 : 0;
            int idx = (y * width + x) * 4;
            rgba[idx + 0] = val;  // R
            rgba[idx + 1] = val;  // G
            rgba[idx + 2] = val;  // B
            rgba[idx + 3] = 255;  // A
        }
    }
}

// Helper: Simple Vulkan device creation for image allocation
static VkDevice createVulkanDevice(VkInstance& instance, VkPhysicalDevice& physDevice) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "pose-warp-test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    // Instance-level capability-query extensions required as prerequisites for
    // VK_KHR_external_memory_win32 on the device (same set as interop-test).
    const char* instExts[] = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount   = static_cast<uint32_t>(std::size(instExts));
    instanceInfo.ppEnabledExtensionNames = instExts;

    if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    physDevice = devices[0];

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = 0;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    const char* extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
    };

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = 2;
    deviceInfo.ppEnabledExtensionNames = extensions;

    VkDevice device;
    if (vkCreateDevice(physDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan device");
    }

    return device;
}

int main() {
    printf("=== Pose Warp Test ===\n");
    printf("Input: %dx%d checkerboard\n", W, H);
    printf("Transform: %.1f dg yaw rotation\n", YAW_DEG);
    printf("Validation: central ROI changed pixels >= %.1f%%\n\n", MIN_CHANGED_PERCENT);

    // -----------------------------------------------------------------------
    // CUDA init
    // -----------------------------------------------------------------------
    CUresult cuErr = cuInit(0);
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "[FAIL] cuInit: %d\n", cuErr);
        return 1;
    }

    CUdevice cuDev;
    cuErr = cuDeviceGet(&cuDev, 0);
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "[FAIL] cuDeviceGet: %d\n", cuErr);
        return 1;
    }

    CUcontext cuCtx;
    cuErr = cuDevicePrimaryCtxRetain(&cuCtx, cuDev);
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "[FAIL] cuDevicePrimaryCtxRetain: %d\n", cuErr);
        return 1;
    }
    cuCtxSetCurrent(cuCtx);

    printf("[OK] CUDA initialized\n");

    // -----------------------------------------------------------------------
    // Vulkan init
    // -----------------------------------------------------------------------
    VkInstance vkInstance = VK_NULL_HANDLE;
    VkPhysicalDevice vkPhysDevice = VK_NULL_HANDLE;
    VkDevice vkDevice = VK_NULL_HANDLE;

    try {
        vkDevice = createVulkanDevice(vkInstance, vkPhysDevice);
        printf("[OK] Vulkan initialized\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "[FAIL] Vulkan init: %s\n", e.what());
        return 1;
    }

    // -----------------------------------------------------------------------
    // Generate input checkerboard
    // -----------------------------------------------------------------------
    std::vector<uint8_t> inputData;
    generateCheckerboard(inputData, W, H);
    printf("[OK] Generated checkerboard pattern\n");

    // -----------------------------------------------------------------------
    // Allocate shared images (Vulkan/CUDA interop)
    // -----------------------------------------------------------------------
    interop::SharedImage inputImage(vkDevice, vkPhysDevice, W, H, VK_FORMAT_R8G8B8A8_UNORM);
    interop::SharedImage outputImage(vkDevice, vkPhysDevice, W, H, VK_FORMAT_R8G8B8A8_UNORM);
    printf("[OK] Allocated shared images\n");

    // -----------------------------------------------------------------------
    // Upload input to CUDA array
    // -----------------------------------------------------------------------
    CUDA_MEMCPY2D copyDesc{};
    copyDesc.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyDesc.srcHost = inputData.data();
    copyDesc.srcPitch = W * 4;
    copyDesc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyDesc.dstArray = inputImage.cuArray();
    copyDesc.WidthInBytes = W * 4;
    copyDesc.Height = H;

    cuErr = cuMemcpy2D(&copyDesc);
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "[FAIL] Upload input: %d\n", cuErr);
        return 1;
    }

    printf("[OK] Uploaded input to GPU\n");

    // -----------------------------------------------------------------------
    // Compute homography
    // -----------------------------------------------------------------------
    // Assume 90° horizontal FOV (symmetric) for test purposes
    const float fovHalfX = 45.0f * 3.14159265f / 180.0f;
    const float fovHalfY = fovHalfX * static_cast<float>(H) / static_cast<float>(W);

    pose_warp::CameraIntrinsics intrinsics = pose_warp::computeIntrinsics(
        fovHalfX, fovHalfX,  // symmetric horizontal FOV
        fovHalfY, fovHalfY,  // symmetric vertical FOV
        W, H
    );

    XrQuaternionf rotation = makeYawRotation(YAW_RAD);
    float homography[9];
    pose_warp::computeRotationHomography(rotation, intrinsics, homography);

    printf("[OK] Computed homography matrix\n");
    printf("     H = [%.4f  %.4f  %.4f]\n", homography[0], homography[1], homography[2]);
    printf("         [%.4f  %.4f  %.4f]\n", homography[3], homography[4], homography[5]);
    printf("         [%.4f  %.4f  %.4f]\n", homography[6], homography[7], homography[8]);

    // -----------------------------------------------------------------------
    // Apply warp
    // -----------------------------------------------------------------------
    try {
        pose_warp::PoseWarper warper;
        warper.warp(inputImage.cuArray(), outputImage.cuArray(), W, H, homography);
        cuCtxSynchronize();
        printf("[OK] Warp applied\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "[FAIL] Warp: %s\n", e.what());
        return 1;
    }

    // -----------------------------------------------------------------------
    // Download output and validate
    // -----------------------------------------------------------------------
    std::vector<uint8_t> outputData(W * H * 4);
    copyDesc.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyDesc.srcArray = outputImage.cuArray();
    copyDesc.dstMemoryType = CU_MEMORYTYPE_HOST;
    copyDesc.dstHost = outputData.data();
    copyDesc.dstPitch = W * 4;

    cuErr = cuMemcpy2D(&copyDesc);
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "[FAIL] Download output: %d\n", cuErr);
        return 1;
    }

    printf("[OK] Downloaded output\n");

    // -----------------------------------------------------------------------
    // Validation: Check that output is not identical to input (warp happened)
    // and that central pixels are reasonable
    // -----------------------------------------------------------------------
    int identicalPixels = 0;
    int totalRoiPixels = 0;

    for (int y = ROI_Y0; y < ROI_Y1; ++y) {
        for (int x = ROI_X0; x < ROI_X1; ++x) {
            int idx = (y * W + x) * 4;
            
            // Check if pixel changed (at least one channel differs by >1)
            bool changed = false;
            for (int c = 0; c < 3; ++c) {
                if (std::abs(static_cast<int>(outputData[idx + c]) - static_cast<int>(inputData[idx + c])) > 1) {
                    changed = true;
                    break;
                }
            }
            
            if (!changed) {
                identicalPixels++;
            }
            
            totalRoiPixels++;
        }
    }

    float percentChanged = 100.0f * (1.0f - static_cast<float>(identicalPixels) / static_cast<float>(totalRoiPixels));
    printf("[INFO] Central ROI: %.1f%% pixels changed\n", percentChanged);

    // For a 5° rotation, we expect significant pixel changes due to shift
    // Accept if at least 20% of central pixels changed
    if (percentChanged < MIN_CHANGED_PERCENT) {
        printf("[FAIL] Too few pixels changed (expected warp effect)\n");
        return 1;
    }

    printf("[PASS] Warp effect detected (%.1f%% changed)\n", percentChanged);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    vkDestroyDevice(vkDevice, nullptr);
    vkDestroyInstance(vkInstance, nullptr);
    cuDevicePrimaryCtxRelease(cuDev);

    printf("\n=== All tests passed ===\n");
    return 0;
}
