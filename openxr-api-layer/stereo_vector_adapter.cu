// MIT License
//
// Copyright(c) 2026 OpenXR Motion Smoothing Layer Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "stereo_vector_adapter.h"

// NOTE: This file uses the CUDA runtime API (cudaMalloc, cudaFree, cudaMemset,
// cudaDeviceSynchronize) while all other subsystems use the CUDA driver API
// (cuMemAlloc, cuMemFree, cuCtxSynchronize).  This inconsistency is intentional
// for the standalone test harness but must be resolved before live integration:
//   - Replace cudaMalloc/cudaFree with cuMemAlloc/cuMemFree.
//   - Replace cudaDeviceSynchronize() with per-stream cuStreamSynchronize().
//   - Update CHECK_CUDA to check CUresult instead of cudaError_t.
// TODO (Item 8 integration): migrate to driver API for consistency.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <cmath>
#include <cstdio>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            char _cudaError[512]; \
            std::snprintf(_cudaError, sizeof(_cudaError), "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(_cudaError); \
        } \
    } while (0)

namespace openxr_api_layer {

// Linearize reversed-Z depth to metric depth in meters.
// Reversed-Z: 1.0 = near plane, 0.0 = far plane (for precision)
// Output: linear depth in [near, far] range
__device__ inline float linearizeDepth(float depth_ndc, float nearPlane, float farPlane) {
    // Reversed-Z formula: depth_linear = (near * far) / (far - depth_ndc * (far - near))
    // This handles the non-linear distribution of depth in NDC space
    return (nearPlane * farPlane) / (farPlane - depth_ndc * (farPlane - nearPlane));
}

// CUDA kernel: Scatter left-eye vectors to right-eye space with depth-tested atomic writes.
// For each left-eye pixel:
//   1. Linearize depth from reversed-Z NDC
//   2. Compute binocular disparity: d = baseline * f_x / depth_linear
//   3. Compute right-eye pixel position: u_R = u_L - d (disparity shifts left)
//   4. Atomically compare-and-swap depth: only write if closer (depth-test)
//   5. If write succeeds, also write the motion vector
//
// @param leftVectors: Input left-eye motion vectors (float2, width×height)
// @param leftDepth: Input left-eye depth (float, width×height, reversed-Z NDC)
// @param rightVectors: Output right-eye motion vectors (float2, width×height), pre-zeroed
// @param rightDepthBuffer: Output/working right-eye depth buffer (float, width×height), pre-initialized to 0.0 (far)
// @param width, height: Image resolution
// @param f_x: Horizontal focal length in pixels
// @param baseline: Interpupillary distance in meters
// @param nearPlane, farPlane: Depth range for linearization
__global__ void kernel_scatter_vectors(
    const float2* leftVectors,
    const float* leftDepth,
    float2* rightVectors,
    float* rightDepthBuffer,
    uint32_t width,
    uint32_t height,
    float f_x,
    float baseline,
    float nearPlane,
    float farPlane
) {
    const uint32_t u_L = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u_L >= width || v >= height) {
        return;
    }

    const uint32_t idx_L = v * width + u_L;
    const float depth_ndc = leftDepth[idx_L];

    // Skip invalid depth (0.0 in reversed-Z means infinite distance)
    if (depth_ndc <= 0.0f) {
        return;
    }

    // Linearize depth from reversed-Z NDC to metric depth
    const float depth_linear = linearizeDepth(depth_ndc, nearPlane, farPlane);

    // Compute binocular disparity in pixels
    // Formula: disparity = baseline * focal_length / depth
    const float disparity = baseline * f_x / depth_linear;

    // Compute right-eye pixel position
    // Disparity shifts pixels to the left in the right eye (negative u offset)
    const float u_R_float = static_cast<float>(u_L) - disparity;
    const int32_t u_R = static_cast<int32_t>(roundf(u_R_float));

    // Check bounds
    if (u_R < 0 || u_R >= static_cast<int32_t>(width)) {
        return;
    }

    const uint32_t idx_R = v * width + u_R;

    // Atomic depth test: only write if this pixel is closer
    // Use atomicCAS (compare-and-swap) to implement depth-testing
    // Reversed-Z: higher values = closer (1.0 = near, 0.0 = far)
    unsigned int* depthAsUint = reinterpret_cast<unsigned int*>(&rightDepthBuffer[idx_R]);
    unsigned int assumedDepth = __float_as_uint(0.0f); // Start with far (0.0 in reversed-Z)
    unsigned int newDepth = __float_as_uint(depth_ndc);

    // Spin until we either successfully write (depth test passes) or fail (occluded)
    bool written = false;
    while (true) {
        unsigned int oldDepth = atomicCAS(depthAsUint, assumedDepth, newDepth);
        
        if (oldDepth == assumedDepth) {
            // Successfully wrote: we are the closest so far
            written = true;
            break;
        }
        
        // Someone else wrote first; check if we are closer
        float oldDepthFloat = __uint_as_float(oldDepth);
        if (depth_ndc > oldDepthFloat) {
            // We are closer (higher value in reversed-Z); try again
            assumedDepth = oldDepth;
        } else {
            // We are farther; give up (occluded by foreground)
            break;
        }
    }

    // If depth test passed, write the motion vector.
    // Note: depth update (atomicCAS) and vector write are not a single atomic transaction.
    // A later, closer writer may update depth and vector after this thread wins CAS, and this
    // thread can still perform its vector store afterward. This is a known scatter-write tradeoff
    // (also present in frame synthesis path) and may cause edge artifacts around disocclusions.
    //
    // The motion vector value is copied unchanged from the left eye. This is correct: the vector
    // represents temporal displacement (frame N → N+1) which is approximately equal for both eyes
    // at normal IPDs (the lateral 6.5 cm offset does not change how scene points move over time).
    // The adapter's job is only to place the vector at the spatially-correct right-eye pixel
    // position (accounting for disparity), not to modify the temporal displacement value.
    if (written) {
        rightVectors[idx_R] = leftVectors[idx_L];
    }
}

// CUDA kernel: Mark holes in the right-eye vector field.
// A pixel is marked as a hole if it has zero depth (no data was scattered to it).
// This is a coverage/no-scatter map, not a full right-eye disocclusion classification.
//
// @param rightDepthBuffer: Right-eye depth buffer after scatter (float, width×height)
// @param holeMap: Output hole map (uint8_t, width×height, 0=valid, 1=hole)
// @param width, height: Image resolution
__global__ void kernel_mark_holes(
    const float* rightDepthBuffer,
    uint8_t* holeMap,
    uint32_t width,
    uint32_t height
) {
    const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) {
        return;
    }

    const uint32_t idx = v * width + u;
    const float depth = rightDepthBuffer[idx];

    // Hole if depth is 0.0 (no scatter wrote to this pixel)
    holeMap[idx] = (depth <= 0.0f) ? 1 : 0;
}

// Constructor: Allocate GPU buffers
StereoVectorAdapter::StereoVectorAdapter(
    uint32_t width,
    uint32_t height,
    float f_x,
    float ipd,
    float nearPlane,
    float farPlane
)
    : m_width(width)
    , m_height(height)
    , m_f_x(f_x)
    , m_ipd(ipd)
    , m_nearPlane(nearPlane)
    , m_farPlane(farPlane)
    , m_rightVectors(nullptr)
    , m_rightDepthBuffer(nullptr)
    , m_holeMap(nullptr)
{
    const size_t numPixels = width * height;
    const size_t vectorBufferSize = numPixels * sizeof(float2);
    const size_t depthBufferSize = numPixels * sizeof(float);
    const size_t holeMapSize = numPixels * sizeof(uint8_t);

    CHECK_CUDA(cudaMalloc(&m_rightVectors, vectorBufferSize));
    CHECK_CUDA(cudaMalloc(&m_rightDepthBuffer, depthBufferSize));
    CHECK_CUDA(cudaMalloc(&m_holeMap, holeMapSize));
}

// Destructor: Free GPU buffers
StereoVectorAdapter::~StereoVectorAdapter() {
    if (m_rightVectors) {
        cudaFree(m_rightVectors);
    }
    if (m_rightDepthBuffer) {
        cudaFree(m_rightDepthBuffer);
    }
    if (m_holeMap) {
        cudaFree(m_holeMap);
    }
}

// Adapt left-eye vectors to right-eye space
StereoVectorAdapter::AdaptResult StereoVectorAdapter::adapt(
    const float2* leftVectors,
    const float* leftDepth,
    const float* rightDepth
) {
    (void)rightDepth;

    const size_t numPixels = m_width * m_height;
    
    // Clear output buffers
    // Right vectors: zero (no motion)
    // Right depth buffer: zero (far in reversed-Z, will be overwritten by scatter)
    // Hole map: will be filled by kernel_mark_holes
    CHECK_CUDA(cudaMemset(m_rightVectors, 0, numPixels * sizeof(float2)));
    CHECK_CUDA(cudaMemset(m_rightDepthBuffer, 0, numPixels * sizeof(float)));

    // Launch scatter kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);

    kernel_scatter_vectors<<<gridSize, blockSize>>>(
        leftVectors,
        leftDepth,
        m_rightVectors,
        m_rightDepthBuffer,
        m_width,
        m_height,
        m_f_x,
        m_ipd, // baseline = IPD
        m_nearPlane,
        m_farPlane
    );
    CHECK_CUDA(cudaGetLastError());

    // Launch hole marking kernel
    kernel_mark_holes<<<gridSize, blockSize>>>(
        m_rightDepthBuffer,
        m_holeMap,
        m_width,
        m_height
    );
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to ensure kernels complete
    CHECK_CUDA(cudaDeviceSynchronize());

    return {m_rightVectors, m_holeMap};
}

} // namespace openxr_api_layer
