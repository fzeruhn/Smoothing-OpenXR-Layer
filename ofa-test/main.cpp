// ofa-test/main.cpp
// Validates OFAPipeline on synthetic input:
//   Frame 0 (reference): 256x256 pseudo-random grayscale noise (seed 0xDEAD)
//   Frame 1 (current):   Frame 0 shifted +8px right, +4px down (zero-padded)
//
// NvOF FORWARD convention: outputBuffer gives displacement from inputFrame →
// referenceFrame (not reference → input).  With inputFrame=frame1 and
// referenceFrame=frame0, the forward flow for an interior block is (-8, -4) px
// = (-256, -128) in S10.5 fixed-point (divide by 32.0f to get pixels).

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ofa_pipeline.h"

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

static constexpr int W       = 256;
static constexpr int H       = 256;
static constexpr int SHIFT_X = 8;   // pixels right (inputFrame is frame0 shifted right)
static constexpr int SHIFT_Y = 4;   // pixels down

// NvOF FORWARD: flow from inputFrame → referenceFrame.
// inputFrame  = frame1 (shifted +8, +4 relative to frame0)
// referenceFrame = frame0 (original)
// Forward flow for interior block: (-8px, -4px) = (-256, -128) in S10.5.
static constexpr int16_t EXPECTED_FX = -256;
static constexpr int16_t EXPECTED_FY = -128;
static constexpr int16_t TOLERANCE   = 2;   // +/-2 S10.5 units = +/-1/16 px

// Pixels in the central 50x50 region of the 64x64 output are validated.
static constexpr int OUT_W    = W / 4;   // 64
static constexpr int OUT_H    = H / 4;   // 64
static constexpr int ROI_SIZE = 50;
static constexpr int ROI_X0   = (OUT_W - ROI_SIZE) / 2;  // 7
static constexpr int ROI_Y0   = (OUT_H - ROI_SIZE) / 2;  // 7

// Simple xorshift32 pseudo-random number generator (reproducible, no stdlib random).
static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int main() {
    // -----------------------------------------------------------------------
    // CUDA init
    // -----------------------------------------------------------------------
    CUresult cuErr = cuInit(0);
    if (cuErr != CUDA_SUCCESS) { fprintf(stderr, "[FAIL] cuInit: %d\n", cuErr); return 1; }

    CUdevice cuDev;
    cuErr = cuDeviceGet(&cuDev, 0);
    if (cuErr != CUDA_SUCCESS) { fprintf(stderr, "[FAIL] cuDeviceGet: %d\n", cuErr); return 1; }

    CUcontext cuCtx;
    cuErr = cuDevicePrimaryCtxRetain(&cuCtx, cuDev);
    if (cuErr != CUDA_SUCCESS) { fprintf(stderr, "[FAIL] cuDevicePrimaryCtxRetain: %d\n", cuErr); return 1; }
    cuCtxSetCurrent(cuCtx);

    // -----------------------------------------------------------------------
    // Synthetic frame generation
    // -----------------------------------------------------------------------
    std::vector<uint8_t> frame0(W * H), frame1(W * H);

    // Frame 0: pseudo-random grayscale noise (fixed seed for reproducibility).
    // Random noise gives every OFA grid block unique texture, avoiding the
    // aperture problem that checkerboard uniform regions cause.
    {
        uint32_t state = 0xDEADBEEFu;
        for (auto& p : frame0)
            p = static_cast<uint8_t>(xorshift32(state) & 0xFF);
    }

    // Frame 1: frame0 shifted +SHIFT_X right, +SHIFT_Y down (zero-padded border)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int srcX = x - SHIFT_X;
            int srcY = y - SHIFT_Y;
            frame1[y * W + x] = (srcX >= 0 && srcX < W && srcY >= 0 && srcY < H)
                                 ? frame0[srcY * W + srcX]
                                 : 0u;
        }
    }

    // -----------------------------------------------------------------------
    // Run OFA
    // -----------------------------------------------------------------------
    bool passed = false;
    try {
        // Nested scope: OFAPipeline must be destroyed before cuDevicePrimaryCtxRelease.
        {
            OFAPipeline ofa(cuCtx, W, H,
                            NV_OF_OUTPUT_VECTOR_GRID_SIZE_4,
                            NV_OF_PERF_LEVEL_MEDIUM);

            // slot 0 = inputFrame (current/new), slot 1 = referenceFrame (previous/old)
            ofa.loadFrame(0, frame1.data());
            ofa.loadFrame(1, frame0.data());
            ofa.execute();

            // Synchronize before reading results.
            CUresult syncErr = cuCtxSynchronize();
            if (syncErr != CUDA_SUCCESS) {
                fprintf(stderr, "[FAIL] cuCtxSynchronize: %d\n", syncErr);
                cuDevicePrimaryCtxRelease(cuDev);
                return 1;
            }

            const NV_OF_FLOW_VECTOR* vectors = ofa.outputData();

            // -----------------------------------------------------------------------
            // Validation: central 50x50 region, >=95% within tolerance
            // -----------------------------------------------------------------------
            int pass_count = 0, total = 0;
            int first_fail_x = -1, first_fail_y = -1;
            int16_t first_fail_fx = 0, first_fail_fy = 0;

            for (int y = ROI_Y0; y < ROI_Y0 + ROI_SIZE; ++y) {
                for (int x = ROI_X0; x < ROI_X0 + ROI_SIZE; ++x) {
                    const NV_OF_FLOW_VECTOR& v = vectors[y * OUT_W + x];
                    bool ok = (std::abs(static_cast<int>(v.flowx) - EXPECTED_FX) <= TOLERANCE) &&
                              (std::abs(static_cast<int>(v.flowy) - EXPECTED_FY) <= TOLERANCE);
                    if (ok) {
                        ++pass_count;
                    } else if (first_fail_x < 0) {
                        first_fail_x = x; first_fail_y = y;
                        first_fail_fx = v.flowx; first_fail_fy = v.flowy;
                    }
                    ++total;
                }
            }

            float pass_pct = 100.0f * pass_count / total;
            passed = (pass_pct >= 95.0f);

            if (passed) {
                printf("[PASS] OFA verified: %d/%d central vectors within tolerance (%.1f%%)\n",
                       pass_count, total, pass_pct);
            } else {
                printf("[FAIL] Only %d/%d vectors within tolerance (%.1f%%)\n",
                       pass_count, total, pass_pct);
                if (first_fail_x >= 0)
                    printf("       First fail at output (%d,%d): flowx=%d (exp %d), flowy=%d (exp %d)\n",
                           first_fail_x, first_fail_y,
                           first_fail_fx, EXPECTED_FX,
                           first_fail_fy, EXPECTED_FY);
            }

            // -----------------------------------------------------------------------
            // PNG output: red = X flow, green = Y flow (+/-16px range -> 0-255)
            // -----------------------------------------------------------------------
            std::vector<uint8_t> img(OUT_W * OUT_H * 3);
            for (int i = 0; i < OUT_W * OUT_H; ++i) {
                float fx = vectors[i].flowx / 32.0f;  // S10.5 -> pixels
                float fy = vectors[i].flowy / 32.0f;
                auto encode = [](float v) -> uint8_t {
                    float norm = (v + 16.0f) / 32.0f;  // map [-16,+16] px -> [0,1]
                    return static_cast<uint8_t>(std::clamp(norm * 255.0f, 0.0f, 255.0f));
                };
                img[i * 3 + 0] = encode(fx);
                img[i * 3 + 1] = encode(fy);
                img[i * 3 + 2] = 0;
            }
            stbi_write_png("ofa-test-output.png", OUT_W, OUT_H, 3, img.data(), OUT_W * 3);
            printf("ofa-test-output.png written (%dx%d)\n", OUT_W, OUT_H);
        } // OFAPipeline destructor runs here — context still valid

    } catch (const std::exception& e) {
        fprintf(stderr, "[FAIL] Exception: %s\n", e.what());
        cuDevicePrimaryCtxRelease(cuDev);
        return 1;
    }

    cuDevicePrimaryCtxRelease(cuDev);
    return passed ? 0 : 1;
}
