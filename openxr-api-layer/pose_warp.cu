// MIT License
// Pose Pre-Warp - CUDA Homography Warping Implementation

#include "pose_warp.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

namespace pose_warp {

// CUDA kernel: Apply homography warp with bilinear interpolation
// Uses backward warping: iterate output pixels, sample input via H^-1
__global__ void kernel_homography_warp(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    uint32_t width,
    uint32_t height,
    float h00, float h01, float h02,
    float h10, float h11, float h12,
    float h20, float h21, float h22
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Apply inverse homography: (u, v, w) = H^-1 * (x, y, 1)
    const float px = static_cast<float>(x);
    const float py = static_cast<float>(y);
    
    const float u = h00 * px + h01 * py + h02;
    const float v = h10 * px + h11 * py + h12;
    const float w = h20 * px + h21 * py + h22;

    // Perspective divide to get source coordinates
    const float w_inv = 1.0f / w;
    const float src_x = u * w_inv;
    const float src_y = v * w_inv;

    // Check bounds
    if (src_x < 0.0f || src_x >= static_cast<float>(width - 1) ||
        src_y < 0.0f || src_y >= static_cast<float>(height - 1)) {
        // Out of bounds: write black
        surf2Dwrite(make_uchar4(0, 0, 0, 0), output, x * sizeof(uchar4), y);
        return;
    }

    // Bilinear interpolation
    const int x0 = static_cast<int>(floorf(src_x));
    const int y0 = static_cast<int>(floorf(src_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float fx = src_x - static_cast<float>(x0);
    const float fy = src_y - static_cast<float>(y0);
    const float fx1 = 1.0f - fx;
    const float fy1 = 1.0f - fy;

    // Read four neighboring pixels
    uchar4 p00, p01, p10, p11;
    surf2Dread(&p00, input, x0 * sizeof(uchar4), y0);
    surf2Dread(&p01, input, x0 * sizeof(uchar4), y1);
    surf2Dread(&p10, input, x1 * sizeof(uchar4), y0);
    surf2Dread(&p11, input, x1 * sizeof(uchar4), y1);

    // Bilinear blend
    const float r = fy1 * (fx1 * p00.x + fx * p10.x) + fy * (fx1 * p01.x + fx * p11.x);
    const float g = fy1 * (fx1 * p00.y + fx * p10.y) + fy * (fx1 * p01.y + fx * p11.y);
    const float b = fy1 * (fx1 * p00.z + fx * p10.z) + fy * (fx1 * p01.z + fx * p11.z);
    const float a = fy1 * (fx1 * p00.w + fx * p10.w) + fy * (fx1 * p01.w + fx * p11.w);

    // Write result
    surf2Dwrite(make_uchar4(
        static_cast<unsigned char>(r + 0.5f),
        static_cast<unsigned char>(g + 0.5f),
        static_cast<unsigned char>(b + 0.5f),
        static_cast<unsigned char>(a + 0.5f)
    ), output, x * sizeof(uchar4), y);
}

// Host-side launcher
static void launch_homography_warp(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    uint32_t width,
    uint32_t height,
    const float H_inv[9],
    CUstream stream
) {
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    kernel_homography_warp<<<grid, block, 0, stream>>>(
        input, output, width, height,
        H_inv[0], H_inv[1], H_inv[2],
        H_inv[3], H_inv[4], H_inv[5],
        H_inv[6], H_inv[7], H_inv[8]
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("kernel_homography_warp launch failed: ") + cudaGetErrorString(err));
    }
}

// Helper: invert 3x3 matrix
static bool invert3x3(const float A[9], float A_inv[9]) {
    const float det = A[0] * (A[4] * A[8] - A[5] * A[7])
                    - A[1] * (A[3] * A[8] - A[5] * A[6])
                    + A[2] * (A[3] * A[7] - A[4] * A[6]);

    constexpr float epsilon = 1e-10f;
    if (fabsf(det) < epsilon) {
        return false;
    }

    const float det_inv = 1.0f / det;

    A_inv[0] = (A[4] * A[8] - A[5] * A[7]) * det_inv;
    A_inv[1] = (A[2] * A[7] - A[1] * A[8]) * det_inv;
    A_inv[2] = (A[1] * A[5] - A[2] * A[4]) * det_inv;

    A_inv[3] = (A[5] * A[6] - A[3] * A[8]) * det_inv;
    A_inv[4] = (A[0] * A[8] - A[2] * A[6]) * det_inv;
    A_inv[5] = (A[2] * A[3] - A[0] * A[5]) * det_inv;

    A_inv[6] = (A[3] * A[7] - A[4] * A[6]) * det_inv;
    A_inv[7] = (A[1] * A[6] - A[0] * A[7]) * det_inv;
    A_inv[8] = (A[0] * A[4] - A[1] * A[3]) * det_inv;

    return true;
}

// PoseWarper implementation
PoseWarper::PoseWarper() {
    // Nothing to initialize at construction time
    // CUDA context must already be current
}

PoseWarper::~PoseWarper() {
    destroy();
}

PoseWarper::PoseWarper(PoseWarper&& other) noexcept {
    // Nothing to move — this is a stateless wrapper
}

PoseWarper& PoseWarper::operator=(PoseWarper&& other) noexcept {
    if (this != &other) {
        destroy();
        // Nothing to move
    }
    return *this;
}

void PoseWarper::destroy() noexcept {
    // No persistent state to clean up
}

void PoseWarper::warp(CUarray input, CUarray output,
                      uint32_t width, uint32_t height,
                      const float homography[9],
                      CUstream stream) {
    // Step 1: Invert homography for backward warping
    float H_inv[9];
    if (!invert3x3(homography, H_inv)) {
        throw std::runtime_error("PoseWarper::warp: homography matrix is singular");
    }

    // Step 2: Create CUDA surface objects
    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    cudaSurfaceObject_t inputSurf = 0;
    cudaSurfaceObject_t outputSurf = 0;

    resDesc.res.array.array = input;
    cudaError_t err = cudaCreateSurfaceObject(&inputSurf, &resDesc);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to create input surface: ") + cudaGetErrorString(err));
    }

    resDesc.res.array.array = output;
    err = cudaCreateSurfaceObject(&outputSurf, &resDesc);
    if (err != cudaSuccess) {
        cudaDestroySurfaceObject(inputSurf);
        throw std::runtime_error(std::string("Failed to create output surface: ") + cudaGetErrorString(err));
    }

    // Step 3: Launch kernel
    try {
        launch_homography_warp(inputSurf, outputSurf, width, height, H_inv, stream);
    } catch (...) {
        cudaDestroySurfaceObject(inputSurf);
        cudaDestroySurfaceObject(outputSurf);
        throw;
    }

    // Step 4: Clean up surface objects
    cudaDestroySurfaceObject(inputSurf);
    cudaDestroySurfaceObject(outputSurf);
}

} // namespace pose_warp
