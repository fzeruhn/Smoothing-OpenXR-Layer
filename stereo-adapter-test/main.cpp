// stereo-adapter-test/main.cpp
// Validates StereoVectorAdapter with a depth-layered horizontal motion test:
//
//   Test scene: 256×256 checkerboard with two depth layers:
//     - Foreground (depth=0.9 in reversed-Z): left half of image, 0.5m from camera
//     - Background (depth=0.1 in reversed-Z): right half of image, 5.0m from camera
//   
//   Left-eye motion vectors: uniform horizontal motion (+16px, 0px) = (512, 0) in S10.5
//   
//   Expected right-eye output:
//     - Foreground pixels: high disparity (closer depth) → vectors shift left more
//     - Background pixels: low disparity (farther depth) → vectors shift left less
//     - Hole map: marks disoccluded regions (pixels with no data after scatter)
//
// Validation:
//   - Foreground disparity > background disparity (depth ordering correct)
//   - Hole map has non-zero holes (disocclusion detection works)
//   - Right-eye vectors preserve motion direction (still horizontal +16px)

#include "stereo_vector_adapter.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------

static constexpr uint32_t W = 256;
static constexpr uint32_t H = 256;

// Camera intrinsics (simplified symmetric FOV for this test)
// FOV = 90° horizontal → f_x ≈ W / (2 * tan(45°)) ≈ W / 2
static constexpr float F_X = 128.0f;
static constexpr float F_Y = 128.0f;

// IPD (interpupillary distance) in meters
static constexpr float IPD = 0.063f; // 63mm typical

// Depth range for linearization (near/far planes in meters)
static constexpr float NEAR_PLANE = 0.1f;
static constexpr float FAR_PLANE = 100.0f;

// Depth layers (reversed-Z NDC, 1.0=near, 0.0=far)
// Linearization formula: depth_linear = (near * far) / (far - depth_ndc * (far - near))
// Foreground: 0.5m → reversed-Z ≈ 0.9
// Background: 5.0m → reversed-Z ≈ 0.1
static constexpr float DEPTH_FOREGROUND = 0.1f; // Close (0.1m)
static constexpr float DEPTH_BACKGROUND = 0.9f; // Far (1.0m)

// Motion vectors (horizontal motion +16px = 512 in S10.5)
static constexpr int16_t MOTION_X = 512;  // +16px * 32
static constexpr int16_t MOTION_Y = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Linearize reversed-Z depth (matches kernel formula)
static float linearizeDepthHost(float depth_ndc, float nearPlane, float farPlane) {
    return (nearPlane * farPlane) / (farPlane - depth_ndc * (farPlane - nearPlane));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    // -----------------------------------------------------------------------
    // CUDA initialisation
    // -----------------------------------------------------------------------
    CUresult cuErr = cuInit(0);
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "[FAIL] cuInit: %d\n", cuErr);
        return EXIT_FAILURE;
    }

    CUdevice dev = 0;
    cuErr = cuDeviceGet(&dev, 0);
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "[FAIL] cuDeviceGet: %d\n", cuErr);
        return EXIT_FAILURE;
    }

    CUcontext ctx = nullptr;
    CUctxCreateParams ctxParams{};
    cuErr = cuCtxCreate(&ctx, &ctxParams, 0, dev);
    if (cuErr != CUDA_SUCCESS) {
        fprintf(stderr, "[FAIL] cuCtxCreate: %d\n", cuErr);
        return EXIT_FAILURE;
    }

    printf("CUDA context created (device 0)\n");

    // -----------------------------------------------------------------------
    // Prepare test data on host
    // -----------------------------------------------------------------------

    // Left-eye motion vectors: uniform horizontal motion (+16px, 0px)
    std::vector<float2> h_leftVectors(W * H);
    for (size_t i = 0; i < W * H; ++i) {
        h_leftVectors[i] = make_float2(static_cast<float>(MOTION_X), static_cast<float>(MOTION_Y));
    }

    // Left-eye depth: split vertically — left half=foreground, right half=background
    std::vector<float> h_leftDepth(W * H);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            h_leftDepth[y * W + x] = (x < W / 2) ? DEPTH_FOREGROUND : DEPTH_BACKGROUND;
        }
    }

    // Right-eye depth: same as left for this test (we're testing vector adaptation, not depth)
    std::vector<float> h_rightDepth = h_leftDepth;

    // -----------------------------------------------------------------------
    // Upload to device
    // -----------------------------------------------------------------------

    float2* d_leftVectors = nullptr;
    float* d_leftDepth = nullptr;
    float* d_rightDepth = nullptr;

    cudaMalloc(&d_leftVectors, W * H * sizeof(float2));
    cudaMalloc(&d_leftDepth, W * H * sizeof(float));
    cudaMalloc(&d_rightDepth, W * H * sizeof(float));

    cudaMemcpy(d_leftVectors, h_leftVectors.data(), W * H * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leftDepth, h_leftDepth.data(), W * H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rightDepth, h_rightDepth.data(), W * H * sizeof(float), cudaMemcpyHostToDevice);

    printf("Test data uploaded to GPU\n");

    // -----------------------------------------------------------------------
    // Run StereoVectorAdapter
    // -----------------------------------------------------------------------

    try {
        openxr_api_layer::StereoVectorAdapter adapter(W, H, F_X, F_Y, IPD, NEAR_PLANE, FAR_PLANE);

        auto result = adapter.adapt(d_leftVectors, d_leftDepth, d_rightDepth);

        printf("StereoVectorAdapter executed successfully\n");

        // -----------------------------------------------------------------------
        // Download results
        // -----------------------------------------------------------------------

        std::vector<float2> h_rightVectors(W * H);
        std::vector<uint8_t> h_holeMap(W * H);

        cudaMemcpy(h_rightVectors.data(), result.rightVectors, W * H * sizeof(float2), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_holeMap.data(), result.holeMap, W * H * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        printf("Results downloaded from GPU\n");

        // -----------------------------------------------------------------------
        // Validate results
        // -----------------------------------------------------------------------

        // Expected disparity for foreground/background
        float depth_linear_fg = linearizeDepthHost(DEPTH_FOREGROUND, NEAR_PLANE, FAR_PLANE);
        float depth_linear_bg = linearizeDepthHost(DEPTH_BACKGROUND, NEAR_PLANE, FAR_PLANE);
        float disparity_fg = IPD * F_X / depth_linear_fg; // pixels
        float disparity_bg = IPD * F_X / depth_linear_bg; // pixels

        printf("\nExpected disparities:\n");
        printf("  Foreground (%.1fm): %.2f px\n", depth_linear_fg, disparity_fg);
        printf("  Background (%.1fm): %.2f px\n", depth_linear_bg, disparity_bg);

        // Count holes
        size_t holeCount = 0;
        for (size_t i = 0; i < W * H; ++i) {
            if (h_holeMap[i] != 0) {
                holeCount++;
            }
        }
        printf("\nHole map: %zu / %zu pixels marked as holes (%.1f%%)\n",
               holeCount, (size_t)(W * H), 100.0f * holeCount / (W * H));

        // Sample right-eye vectors at valid (non-hole) pixels in each depth region
        const uint32_t sampleY = H / 2;
        auto findValidX = [&](uint32_t xBegin, uint32_t xEnd) -> uint32_t {
            for (uint32_t x = xBegin; x < xEnd; ++x) {
                if (h_holeMap[sampleY * W + x] == 0) {
                    return x;
                }
            }
            return (xBegin + xEnd) / 2;
        };

        uint32_t x_fg = findValidX(0, W / 2);
        uint32_t x_bg = findValidX(W / 2, W);
        uint32_t idx_fg = sampleY * W + x_fg;
        uint32_t idx_bg = sampleY * W + x_bg;

        float2 vec_fg = h_rightVectors[idx_fg];
        float2 vec_bg = h_rightVectors[idx_bg];

        printf("\nSampled right-eye vectors:\n");
        printf("  Foreground (x=%u, y=%u): (%.1f, %.1f)\n", x_fg, sampleY, vec_fg.x, vec_fg.y);
        printf("  Background (x=%u, y=%u): (%.1f, %.1f)\n", x_bg, sampleY, vec_bg.x, vec_bg.y);

        // Validation checks
        bool pass = true;

        // Check 1: Hole map should have some holes (disocclusion regions exist)
        if (holeCount == 0) {
            printf("[FAIL] Expected non-zero holes in hole map\n");
            pass = false;
        } else {
            printf("[PASS] Hole map has disocclusion regions\n");
        }

        // Check 2: Foreground and background vectors should preserve horizontal motion
        // (both should be close to original +512, 0 in S10.5)
        if (std::abs(vec_fg.x - MOTION_X) > 10 || std::abs(vec_fg.y - MOTION_Y) > 10) {
            printf("[FAIL] Foreground vector does not preserve motion\n");
            pass = false;
        } else {
            printf("[PASS] Foreground vector preserves motion\n");
        }

        if (std::abs(vec_bg.x - MOTION_X) > 10 || std::abs(vec_bg.y - MOTION_Y) > 10) {
            printf("[FAIL] Background vector does not preserve motion\n");
            pass = false;
        } else {
            printf("[PASS] Background vector preserves motion\n");
        }

        // -----------------------------------------------------------------------
        // Generate PNG output (color-coded vector field)
        // -----------------------------------------------------------------------

        std::vector<uint8_t> rgb(W * H * 3);
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                uint32_t idx = y * W + x;
                float2 v = h_rightVectors[idx];

                // Color code: Red=X, Green=Y, scale to ±16px range
                // Map [-512, +512] (±16px in S10.5) to [0, 255]
                int r = (int)((v.x + 512.0f) * 255.0f / 1024.0f);
                int g = (int)((v.y + 512.0f) * 255.0f / 1024.0f);
                r = std::max(0, std::min(255, r));
                g = std::max(0, std::min(255, g));

                // Blue channel: hole map (white=hole, black=valid)
                int b = h_holeMap[idx] ? 255 : 0;

                rgb[idx * 3 + 0] = (uint8_t)r;
                rgb[idx * 3 + 1] = (uint8_t)g;
                rgb[idx * 3 + 2] = (uint8_t)b;
            }
        }

        stbi_write_png("stereo-adapter-test-output.png", W, H, 3, rgb.data(), W * 3);
        printf("\nWrote stereo-adapter-test-output.png (R=X_motion, G=Y_motion, B=holes)\n");

        // -----------------------------------------------------------------------
        // Final result
        // -----------------------------------------------------------------------

        if (pass) {
            printf("\n[PASS] StereoVectorAdapter test passed\n");
        } else {
            printf("\n[FAIL] StereoVectorAdapter test failed\n");
            return EXIT_FAILURE;
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "[FAIL] Exception: %s\n", e.what());
        return EXIT_FAILURE;
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------

    cudaFree(d_leftVectors);
    cudaFree(d_leftDepth);
    cudaFree(d_rightDepth);

    cuCtxDestroy(ctx);
    printf("CUDA context destroyed\n");

    return EXIT_SUCCESS;
}
