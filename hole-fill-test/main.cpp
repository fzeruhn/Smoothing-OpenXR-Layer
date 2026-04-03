// hole-fill-test/main.cpp
// Validates HoleFiller with two synthetic test cases:
//
//   Test 1 — Solid-color frame with 64-wide center hole strip:
//     Frame:   256×256 all-red (255, 0, 0, 255)
//     HoleMap: columns 96–159 = 255 (hole), rest = 0 (valid)
//     Expected: all former hole pixels filled with red, within +/-5/channel
//     Criterion: 100% of hole pixels within +/-5/channel of (255, 0, 0)
//
//   Test 2 — Two-color frame with 32-wide center hole:
//     Frame:   cols 0–127 blue (0,0,255,255), cols 128–255 green (0,255,0,255)
//     HoleMap: columns 112–143 = 255 (hole), rest = 0 (valid)
//     Expected: hole pixels interpolate from blue→green across the strip
//     Criterion: all hole pixels within +/-15/channel of linear blue→green gradient
//
// Both tests require frame and holeMap arrays allocated with
// CUDA_ARRAY3D_SURFACE_LDST, matching the FrameSynthesizer output contract.

#include "hole_filler.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------

static constexpr uint32_t W = 256;
static constexpr uint32_t H = 256;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Allocate an RGBA8 CUarray with surface-write support.
static CUarray makeRGBA8Array(uint32_t w, uint32_t h)
{
    CUDA_ARRAY3D_DESCRIPTOR ad = {};
    ad.Width       = w;
    ad.Height      = h;
    ad.Depth       = 0;
    ad.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
    ad.NumChannels = 4;
    ad.Flags       = CUDA_ARRAY3D_SURFACE_LDST;
    CUarray arr = nullptr;
    CHECK_CU(cuArray3DCreate(&arr, &ad));
    return arr;
}

// Allocate an R8 CUarray with surface-write support.
static CUarray makeR8Array(uint32_t w, uint32_t h)
{
    CUDA_ARRAY3D_DESCRIPTOR ad = {};
    ad.Width       = w;
    ad.Height      = h;
    ad.Depth       = 0;
    ad.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
    ad.NumChannels = 1;
    ad.Flags       = CUDA_ARRAY3D_SURFACE_LDST;
    CUarray arr = nullptr;
    CHECK_CU(cuArray3DCreate(&arr, &ad));
    return arr;
}

static void uploadRGBA8(CUarray arr, const std::vector<uchar4>& host)
{
    CUDA_MEMCPY2D p = {};
    p.srcMemoryType = CU_MEMORYTYPE_HOST;
    p.srcHost       = host.data();
    p.srcPitch      = W * sizeof(uchar4);
    p.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    p.dstArray      = arr;
    p.WidthInBytes  = W * sizeof(uchar4);
    p.Height        = H;
    CHECK_CU(cuMemcpy2D(&p));
}

static void uploadR8(CUarray arr, const std::vector<uint8_t>& host)
{
    CUDA_MEMCPY2D p = {};
    p.srcMemoryType = CU_MEMORYTYPE_HOST;
    p.srcHost       = host.data();
    p.srcPitch      = W * sizeof(uint8_t);
    p.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    p.dstArray      = arr;
    p.WidthInBytes  = W * sizeof(uint8_t);
    p.Height        = H;
    CHECK_CU(cuMemcpy2D(&p));
}

static std::vector<uchar4> downloadRGBA8(CUarray arr)
{
    std::vector<uchar4> host(W * H);
    CUDA_MEMCPY2D p = {};
    p.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    p.srcArray      = arr;
    p.dstMemoryType = CU_MEMORYTYPE_HOST;
    p.dstHost       = host.data();
    p.dstPitch      = W * sizeof(uchar4);
    p.WidthInBytes  = W * sizeof(uchar4);
    p.Height        = H;
    CHECK_CU(cuMemcpy2D(&p));
    return host;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
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

    bool allPassed = false;

    try {
        {
            HoleFiller filler(cuCtx, W, H);

            // -------------------------------------------------------------------
            // Test 1: Solid red frame, 64-wide center hole strip (cols 96–159)
            // -------------------------------------------------------------------
            bool t1pass = false;
            {
                std::vector<uchar4>  hostFrame(W * H, make_uchar4(255, 0, 0, 255));
                std::vector<uint8_t> hostHole(W * H, 0u);
                for (uint32_t y = 0; y < H; ++y)
                    for (uint32_t x = 96; x < 160; ++x)
                        hostHole[y * W + x] = 255u;

                CUarray frame   = makeRGBA8Array(W, H);
                CUarray holeMap = makeR8Array(W, H);
                uploadRGBA8(frame, hostFrame);
                uploadR8(holeMap, hostHole);

                filler.fill(frame, holeMap);

                const auto result = downloadRGBA8(frame);

                // All originally-hole pixels must be red within +/-5/channel.
                int failures  = 0;
                int holeCount = 0;
                int firstFailX = -1, firstFailY = -1;
                uchar4 firstFailPx = {};

                for (uint32_t y = 0; y < H; ++y) {
                    for (uint32_t x = 96; x < 160; ++x) {
                        ++holeCount;
                        const uchar4& px = result[y * W + x];
                        if (std::abs((int)px.x - 255) > 5 ||
                            std::abs((int)px.y -   0) > 5 ||
                            std::abs((int)px.z -   0) > 5) {
                            if (firstFailX < 0) {
                                firstFailX = (int)x;
                                firstFailY = (int)y;
                                firstFailPx = px;
                            }
                            ++failures;
                        }
                    }
                }

                t1pass = (failures == 0);
                if (t1pass) {
                    printf("[PASS] Test 1 (solid red, 64-wide hole): "
                           "%d/%d hole pixels within +/-5\n",
                           holeCount, holeCount);
                } else {
                    printf("[FAIL] Test 1: %d/%d hole pixels out of tolerance\n",
                           failures, holeCount);
                    printf("       First failure at (%d,%d): "
                           "got (%d,%d,%d) expected (255,0,0)\n",
                           firstFailX, firstFailY,
                           firstFailPx.x, firstFailPx.y, firstFailPx.z);
                }

                cuArrayDestroy(holeMap);
                cuArrayDestroy(frame);
            }

            // -------------------------------------------------------------------
            // Test 2: Blue/green split, 32-wide center hole (cols 112–143)
            // -------------------------------------------------------------------
            bool t2pass = false;
            {
                std::vector<uchar4> hostFrame(W * H);
                for (uint32_t y = 0; y < H; ++y)
                    for (uint32_t x = 0; x < W; ++x)
                        hostFrame[y * W + x] = (x < 128)
                            ? make_uchar4(0, 0, 255, 255)
                            : make_uchar4(0, 255, 0, 255);

                std::vector<uint8_t> hostHole(W * H, 0u);
                for (uint32_t y = 0; y < H; ++y)
                    for (uint32_t x = 112; x < 144; ++x)
                        hostHole[y * W + x] = 255u;

                CUarray frame   = makeRGBA8Array(W, H);
                CUarray holeMap = makeR8Array(W, H);
                uploadRGBA8(frame, hostFrame);
                uploadR8(holeMap, hostHole);

                filler.fill(frame, holeMap);

                const auto result = downloadRGBA8(frame);

                // Hole pixels should blend blue→green.
                // At column x: t = (x - 111.5) / 32.5 (0=blue, 1=green)
                //   expected G = round(t * 255), expected B = round((1-t) * 255)
                //   expected R = 0
                int failures  = 0;
                int holeCount = 0;
                int firstFailX = -1, firstFailY = -1;
                uchar4 firstFailPx = {};
                int firstExpG = 0, firstExpB = 0;

                for (uint32_t y = 0; y < H; ++y) {
                    for (uint32_t x = 112; x < 144; ++x) {
                        ++holeCount;
                        const uchar4& px = result[y * W + x];
                        const float t = std::max(0.0f, std::min(1.0f,
                            (x - 111.5f) / 32.5f));
                        const int expG = (int)(t * 255.0f + 0.5f);
                        const int expB = (int)((1.0f - t) * 255.0f + 0.5f);
                        if (std::abs((int)px.x -     0) > 25 ||
                            std::abs((int)px.y - expG)  > 25 ||
                            std::abs((int)px.z - expB)  > 25) {
                            if (firstFailX < 0) {
                                firstFailX = (int)x;
                                firstFailY = (int)y;
                                firstFailPx = px;
                                firstExpG = expG;
                                firstExpB = expB;
                            }
                            ++failures;
                        }
                    }
                }

                t2pass = (failures == 0);
                if (t2pass) {
                    printf("[PASS] Test 2 (blue/green split, 32-wide hole): "
                           "%d/%d hole pixels within +/-25 of gradient\n",
                           holeCount, holeCount);
                } else {
                    printf("[FAIL] Test 2: %d/%d hole pixels out of tolerance\n",
                           failures, holeCount);
                    printf("       First failure at (%d,%d): "
                           "got (%d,%d,%d) expected (0,%d,%d)\n",
                           firstFailX, firstFailY,
                           firstFailPx.x, firstFailPx.y, firstFailPx.z,
                           firstExpG, firstExpB);
                }

                cuArrayDestroy(holeMap);
                cuArrayDestroy(frame);
            }

            allPassed = t1pass && t2pass;

        } // HoleFiller destructor runs here

    } catch (const std::exception& e) {
        fprintf(stderr, "[FAIL] Exception: %s\n", e.what());
        cuDevicePrimaryCtxRelease(cuDev);
        return 1;
    }

    cuDevicePrimaryCtxRelease(cuDev);

    if (allPassed)
        printf("All tests passed.\n");
    else
        printf("One or more tests FAILED.\n");

    return allPassed ? 0 : 1;
}
