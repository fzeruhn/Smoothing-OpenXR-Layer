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

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace openxr_api_layer {

// RAII wrapper for deriving right-eye motion vectors from left-eye OFA output.
// Uses binocular geometry: for each left-eye pixel, computes disparity from depth,
// shifts the vector to the right-eye coordinate frame, and performs atomic depth-testing
// to handle occlusions (closest surface wins).
class StereoVectorAdapter {
public:
    // Constructor: allocates GPU buffers for the right-eye vector field, depth buffer, and hole map.
    // @param width, height: resolution of the left-eye input (right-eye assumed same resolution)
    // @param f_x: horizontal focal length in pixels from asymmetric FOV
    // @param ipd: interpupillary distance in meters (distance between eyes)
    // @param nearPlane, farPlane: depth range for linearization (from projection matrix)
    StereoVectorAdapter(uint32_t width, uint32_t height, float f_x, float ipd, float nearPlane, float farPlane);
    ~StereoVectorAdapter();

    StereoVectorAdapter(const StereoVectorAdapter&) = delete;
    StereoVectorAdapter& operator=(const StereoVectorAdapter&) = delete;

    // Adapt left-eye motion vectors to right-eye space.
    // @param leftVectors: GPU buffer of left-eye motion vectors (float2, width×height)
    // @param leftDepth: GPU buffer of left-eye depth values (float, width×height, reversed-Z NDC [0,1])
    // @param rightDepth: GPU buffer of right-eye depth values (float, width×height, reversed-Z NDC [0,1])
    //                    reserved for future depth-consistency/occlusion logic; currently unused.
    // @return: tuple of (rightVectors, holeMap)
    //   - rightVectors: GPU buffer of adapted right-eye motion vectors (float2, width×height)
    //   - holeMap: GPU buffer marking no-scatter coverage gaps (uint8_t, width×height, 0=covered, 1=uncovered)
    //              This is not a full disocclusion map yet; future revisions will incorporate rightDepth.
    // Note: Outputs are owned by StereoVectorAdapter and reused across calls; do not free.
    struct AdaptResult {
        float2* rightVectors;
        uint8_t* holeMap;
    };
    AdaptResult adapt(const float2* leftVectors, const float* leftDepth, const float* rightDepth);

    // Accessors for internal buffers (for testing/debugging)
    float2* getRightVectors() const { return m_rightVectors; }
    uint8_t* getHoleMap() const { return m_holeMap; }

private:
    uint32_t m_width;
    uint32_t m_height;
    float m_f_x;
    float m_ipd;
    float m_nearPlane;
    float m_farPlane;

    // GPU buffers (managed by this class)
    float2* m_rightVectors; // Right-eye motion vectors (device memory)
    float* m_rightDepthBuffer; // Right-eye depth buffer for atomic testing (device memory)
    uint8_t* m_holeMap; // Hole map (0=valid, 1=hole) (device memory)
};

} // namespace openxr_api_layer
