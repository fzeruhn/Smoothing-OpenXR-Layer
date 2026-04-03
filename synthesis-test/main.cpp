// synthesis-test/main.cpp
// Validates FrameSynthesizer with a known rigid-translation test case:
//
//   Frame N   : 256×256 RGBA8 checkerboard (8×8 px black/white tiles)
//   Frame N+1 : Frame N shifted +16 px right, +8 px down (zero-padded border)
//   Vectors   : uniform (+16, +8) px = (512, 256) in S10.5 for every grid cell
//   Expected  : T+0.5 output = Frame N shifted +8 px right, +4 px down
//
// With an integer-pixel shift and uniform vectors the synthesis is exact:
// both the forward scatter (N→T+0.5) and backward scatter (N+1→T+0.5) land
// on the same destination, and the bilinear gathers from both frames return
// identical values, so the 50/50 blend is equal to the expected pixel.
//
// Validation: central ROI [16..239, 16..239], ≥95% of pixels within ±2/channel.
// Hole map: must be all-zero in the central ROI.

#include "frame_synthesizer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------

static constexpr uint32_t W        = 256;
static constexpr uint32_t H        = 256;
static constexpr uint32_t GRID     = 4;    // OFA grid size
static constexpr int      SHIFT_X  = 16;  // N+1 is N shifted +16 px right
static constexpr int      SHIFT_Y  = 8;   // N+1 is N shifted +8  px down

// Expected T+0.5 is N shifted by half the above
static constexpr int HALF_X = SHIFT_X / 2;  // 8
static constexpr int HALF_Y = SHIFT_Y / 2;  // 4

// S10.5: multiply pixel displacement by 32 to get fixed-point units
static constexpr int16_t VEC_FX = (int16_t)(SHIFT_X * 32);  // 512
static constexpr int16_t VEC_FY = (int16_t)(SHIFT_Y * 32);  // 256

// Validation region — avoid the HALF_X/HALF_Y border where the shift clips
static constexpr int ROI_X0 = 16, ROI_X1 = 240;   // [16, 240)
static constexpr int ROI_Y0 = 16, ROI_Y1 = 240;   // [16, 240)

static constexpr int TOLERANCE    = 2;    // per-channel, out of 255
static constexpr float PASS_PCT   = 95.0f;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Returns the RGBA8 value for checkerboard pixel (x,y): 8×8 black/white tiles.
static uchar4 checkerPixel(int x, int y)
{
    uint8_t v = (((x / 8) + (y / 8)) & 1) ? 255u : 0u;
    return make_uchar4(v, v, v, 255u);
}

// Allocate a host-side RGBA8 buffer and fill with the checkerboard.
static std::vector<uchar4> makeCheckerboard()
{
    std::vector<uchar4> buf(W * H);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
            buf[y * W + x] = checkerPixel((int)x, (int)y);
    return buf;
}

// Shift frame by (+dx, +dy), zero-pad the new border.
static std::vector<uchar4> shiftFrame(const std::vector<uchar4>& src, int dx, int dy)
{
    std::vector<uchar4> dst(W * H, make_uchar4(0, 0, 0, 255u));
    for (int y = 0; y < (int)H; ++y) {
        for (int x = 0; x < (int)W; ++x) {
            int sx = x - dx;
            int sy = y - dy;
            if (sx >= 0 && sx < (int)W && sy >= 0 && sy < (int)H)
                dst[y * W + x] = src[sy * W + sx];
        }
    }
    return dst;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    // -----------------------------------------------------------------------
    // CUDA initialisation (same pattern as ofa-test)
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

    bool passed = false;

    try {
        // Nested scope: FrameSynthesizer + device arrays must be destroyed
        // before cuDevicePrimaryCtxRelease.
        {
            // -------------------------------------------------------------------
            // Synthetic frame generation
            // -------------------------------------------------------------------
            const auto hostN   = makeCheckerboard();
            const auto hostN1  = shiftFrame(hostN, SHIFT_X, SHIFT_Y);

            // -------------------------------------------------------------------
            // Upload frame N and N+1 to RGBA8 CUDA arrays
            // -------------------------------------------------------------------
            cudaChannelFormatDesc rgba8 =
                cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

            cudaArray_t arrN = nullptr, arrN1 = nullptr;
            cudaMallocArray(&arrN,  &rgba8, W, H);
            cudaMallocArray(&arrN1, &rgba8, W, H);

            // cudaMemcpy2DToArray: srcPitch = W * sizeof(uchar4) (tightly packed)
            cudaMemcpy2DToArray(arrN,  0, 0, hostN.data(),  W * sizeof(uchar4),
                                W * sizeof(uchar4), H, cudaMemcpyHostToDevice);
            cudaMemcpy2DToArray(arrN1, 0, 0, hostN1.data(), W * sizeof(uchar4),
                                W * sizeof(uchar4), H, cudaMemcpyHostToDevice);

            // -------------------------------------------------------------------
            // Build uniform motion vector grid: all cells = (+SHIFT_X, +SHIFT_Y)
            // in S10.5 format (NV_OF_FLOW_VECTOR compatible).
            // -------------------------------------------------------------------
            const uint32_t gridW = (W + GRID - 1) / GRID;  // 64
            const uint32_t gridH = (H + GRID - 1) / GRID;  // 64
            std::vector<SynthFlowVector> vecs(gridW * gridH, { VEC_FX, VEC_FY });

            // -------------------------------------------------------------------
            // Construct and run FrameSynthesizer
            // -------------------------------------------------------------------
            FrameSynthesizer synth(cuCtx, W, H, GRID);

            synth.loadFrameN ((CUarray)arrN,  nullptr);  // depth=null → uniform 0.5
            synth.loadFrameN1((CUarray)arrN1, nullptr);
            synth.loadMotionVectors(vecs.data(), vecs.size());
            synth.execute();

            // -------------------------------------------------------------------
            // Read back synthesised frame and hole map to host
            // -------------------------------------------------------------------
            std::vector<uchar4>  hostSynth(W * H);
            std::vector<uint8_t> hostHoles(W * H);

            {
                CUDA_MEMCPY2D p = {};
                p.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                p.srcArray      = synth.synthesizedFrame();
                p.dstMemoryType = CU_MEMORYTYPE_HOST;
                p.dstHost       = hostSynth.data();
                p.dstPitch      = W * sizeof(uchar4);
                p.WidthInBytes  = W * sizeof(uchar4);
                p.Height        = H;
                CHECK_CU(cuMemcpy2D(&p));
            }
            {
                CUDA_MEMCPY2D p = {};
                p.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                p.srcArray      = synth.holeMap();
                p.dstMemoryType = CU_MEMORYTYPE_HOST;
                p.dstHost       = hostHoles.data();
                p.dstPitch      = W * sizeof(uint8_t);
                p.WidthInBytes  = W * sizeof(uint8_t);
                p.Height        = H;
                CHECK_CU(cuMemcpy2D(&p));
            }

            // -------------------------------------------------------------------
            // Validation — central ROI
            // -------------------------------------------------------------------

            // 1. Pixel accuracy: each output pixel at (ox, oy) should match
            //    the checkerboard at (ox - HALF_X, oy - HALF_Y).
            int  passCount = 0, total = 0;
            int  firstFailX = -1, firstFailY = -1;
            uchar4 firstFailActual = {}, firstFailExpected = {};

            for (int oy = ROI_Y0; oy < ROI_Y1; ++oy) {
                for (int ox = ROI_X0; ox < ROI_X1; ++ox) {
                    const uchar4 actual   = hostSynth[oy * W + ox];
                    const uchar4 expected = checkerPixel(ox - HALF_X, oy - HALF_Y);

                    const bool ok =
                        std::abs((int)actual.x - (int)expected.x) <= TOLERANCE &&
                        std::abs((int)actual.y - (int)expected.y) <= TOLERANCE &&
                        std::abs((int)actual.z - (int)expected.z) <= TOLERANCE;

                    if (ok) {
                        ++passCount;
                    } else if (firstFailX < 0) {
                        firstFailX = ox; firstFailY = oy;
                        firstFailActual   = actual;
                        firstFailExpected = expected;
                    }
                    ++total;
                }
            }

            const float passPct = 100.0f * (float)passCount / (float)total;
            const bool pixelPass = (passPct >= PASS_PCT);

            // 2. Hole map: must be all-zero in the central ROI.
            int holeCount = 0;
            for (int oy = ROI_Y0; oy < ROI_Y1; ++oy)
                for (int ox = ROI_X0; ox < ROI_X1; ++ox)
                    if (hostHoles[oy * W + ox] != 0) ++holeCount;
            const bool holePass = (holeCount == 0);

            passed = pixelPass && holePass;

            if (passed) {
                printf("[PASS] Bidirectional synthesis verified "
                       "(256x256, checkerboard +%d/+%d translation, T+0.5)\n",
                       SHIFT_X, SHIFT_Y);
                printf("       Pixels: %d/%d within tolerance (%.1f%%), "
                       "Holes in ROI: %d\n",
                       passCount, total, passPct, holeCount);
            } else {
                printf("[FAIL] synthesis-test\n");
                if (!pixelPass) {
                    printf("       Pixel accuracy: only %d/%d within tolerance (%.1f%%), "
                           "need %.0f%%\n",
                           passCount, total, passPct, PASS_PCT);
                    if (firstFailX >= 0)
                        printf("       First fail at (%d,%d): got (%d,%d,%d) "
                               "expected (%d,%d,%d)\n",
                               firstFailX, firstFailY,
                               firstFailActual.x,   firstFailActual.y,   firstFailActual.z,
                               firstFailExpected.x, firstFailExpected.y, firstFailExpected.z);
                }
                if (!holePass)
                    printf("       Hole map: %d unexpected holes in central ROI\n",
                           holeCount);
            }

            // Cleanup device arrays
            cudaFreeArray(arrN1);
            cudaFreeArray(arrN);

        } // FrameSynthesizer destructor runs here

    } catch (const std::exception& e) {
        fprintf(stderr, "[FAIL] Exception: %s\n", e.what());
        cuDevicePrimaryCtxRelease(cuDev);
        return 1;
    }

    cuDevicePrimaryCtxRelease(cuDev);
    return passed ? 0 : 1;
}
