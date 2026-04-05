// MIT License
// Pose Pre-Warp - CUDA Homography Warping

#pragma once

#include <cuda.h>
#include <stdexcept>
#include <string>

namespace pose_warp {

// RAII wrapper for pose-based homography warping pipeline.
//
// Applies a 3x3 homography matrix to warp an image, compensating for
// head rotation between render-time and display-time poses.
//
// Prerequisites:
//   - CUDA must be initialized (cuInit called, context current)
//   - Input and output CUarrays must be allocated and valid
class PoseWarper {
  public:
    PoseWarper();
    ~PoseWarper();

    PoseWarper(const PoseWarper&) = delete;
    PoseWarper& operator=(const PoseWarper&) = delete;
    PoseWarper(PoseWarper&&) noexcept;
    PoseWarper& operator=(PoseWarper&&) noexcept;

    // Apply homography warp to input image
    //
    // Parameters:
    //   input: Source CUarray (RGBA8 or similar format)
    //   output: Destination CUarray (same format as input)
    //   width: Image width in pixels
    //   height: Image height in pixels
    //   homography: Row-major 3x3 homography matrix
    //   stream: CUDA stream for async execution (0 for default stream)
    //
    // Warp type: Backward warp (iterate output pixels, sample input via inverse homography)
    // Sampling: Bilinear interpolation for sub-pixel accuracy
    // Out-of-bounds: Pixels outside input image are set to black (0,0,0,0)
    //
    // IMPORTANT: warp() enqueues the kernel asynchronously and returns immediately.
    // Kernel execution errors (e.g. invalid surface access) will NOT be surfaced here —
    // they appear on the next synchronizing call. The caller must issue
    // cuStreamSynchronize(stream) or cuCtxSynchronize() before reading output pixels.
    void warp(CUarray input, CUarray output,
              uint32_t width, uint32_t height,
              const float homography[9],
              CUstream stream = nullptr);

  private:
    void destroy() noexcept;
};

} // namespace pose_warp
