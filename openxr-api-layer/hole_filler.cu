// hole_filler.cu
// Hierarchical push-pull hole filling — CUDA kernels and HoleFiller implementation.
//
// Four kernel stages run sequentially on the default stream:
//
//   kernel_copy_level0  — copies caller's frame+holeMap into pyramid[0],
//                         inverting holeMap convention (0=valid→255=valid).
//   kernel_push         — downsamples one pyramid level, averaging valid pixels.
//   kernel_pull         — fills hole pixels at one level from the one-coarser
//                         level via bilinear texture sampling.
//   kernel_writeback    — writes filled pyramid[0] colours back to `frame`
//                         at only the originally-hole pixels.
//
// Surface coordinate convention (matches frame_synthesizer.cu):
//   RGBA8 arrays: byte offset = pixel_x * 4
//   R8 arrays:    byte offset = pixel_x

#include "hole_filler.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

// ---------------------------------------------------------------------------
// File-local helpers — surface and texture object creation
// ---------------------------------------------------------------------------

static CUsurfObject makeSurf(CUarray arr)
{
    CUDA_RESOURCE_DESC rd = {};
    rd.resType          = CU_RESOURCE_TYPE_ARRAY;
    rd.res.array.hArray = arr;
    CUsurfObject s = 0;
    CHECK_CU(cuSurfObjectCreate(&s, &rd));
    return s;
}

// Bilinear RGBA8 texture — returns float4 in [0, 1].
static CUtexObject makeTexRGBA8(CUarray arr)
{
    CUDA_RESOURCE_DESC rd = {};
    rd.resType          = CU_RESOURCE_TYPE_ARRAY;
    rd.res.array.hArray = arr;

    CUDA_TEXTURE_DESC td = {};
    td.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    td.filterMode     = CU_TR_FILTER_MODE_LINEAR;
    td.flags          = 0;   // normalised float output

    CUtexObject t = 0;
    CHECK_CU(cuTexObjectCreate(&t, &rd, &td, nullptr));
    return t;
}

// ---------------------------------------------------------------------------
// Stage 1 — kernel_copy_level0
//
// Copies frame and holeMap into pyramid[0], converting the holeMap convention:
//   input  holeMap: 0=valid, 255=hole
//   output valid:   255=valid, 0=hole
// Hole pixels get black colour; valid pixels keep their original colour.
// ---------------------------------------------------------------------------

__global__ void kernel_copy_level0(
    cudaSurfaceObject_t srcFrame,
    cudaSurfaceObject_t srcHoleMap,
    cudaSurfaceObject_t dstColor,
    cudaSurfaceObject_t dstValid,
    uint32_t width, uint32_t height)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint8_t hole;
    surf2Dread(&hole, srcHoleMap, (int)x, (int)y);
    const uint8_t validVal = (hole == 0u) ? 255u : 0u;

    uchar4 color;
    if (validVal) {
        surf2Dread(&color, srcFrame, (int)(x * 4u), (int)y);
    } else {
        color = make_uchar4(0, 0, 0, 0);
    }

    surf2Dwrite(color,    dstColor, (int)(x * 4u), (int)y);
    surf2Dwrite(validVal, dstValid, (int)x,         (int)y);
}

// ---------------------------------------------------------------------------
// Stage 2 — kernel_push
//
// Downsamples one pyramid level.  For each output pixel, accumulates a
// weighted average of its 2×2 source block, weighting only valid pixels.
// Output pixels with no valid source are marked as holes.
// ---------------------------------------------------------------------------

__global__ void kernel_push(
    cudaSurfaceObject_t srcColor,
    cudaSurfaceObject_t srcValid,
    cudaSurfaceObject_t dstColor,
    cudaSurfaceObject_t dstValid,
    uint32_t srcW, uint32_t srcH,
    uint32_t dstW, uint32_t dstH)
{
    const uint32_t dx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float  w   = 0.0f;

    for (int ky = 0; ky < 2; ++ky) {
        for (int kx = 0; kx < 2; ++kx) {
            const uint32_t sx = dx * 2u + (uint32_t)kx;
            const uint32_t sy = dy * 2u + (uint32_t)ky;
            if (sx >= srcW || sy >= srcH) continue;

            uint8_t v;
            surf2Dread(&v, srcValid, (int)sx, (int)sy);
            if (v == 0u) continue;   // hole pixel — skip

            uchar4 c;
            surf2Dread(&c, srcColor, (int)(sx * 4u), (int)sy);
            sum.x += (float)c.x;
            sum.y += (float)c.y;
            sum.z += (float)c.z;
            sum.w += (float)c.w;
            w     += 1.0f;
        }
    }

    if (w > 0.0f) {
        const float inv = 1.0f / w;
        const uchar4 out = make_uchar4(
            (uint8_t)(sum.x * inv + 0.5f),
            (uint8_t)(sum.y * inv + 0.5f),
            (uint8_t)(sum.z * inv + 0.5f),
            (uint8_t)(sum.w * inv + 0.5f));
        surf2Dwrite(out,           dstColor, (int)(dx * 4u), (int)dy);
        surf2Dwrite((uint8_t)255u, dstValid, (int)dx,         (int)dy);
    } else {
        surf2Dwrite(make_uchar4(0, 0, 0, 0), dstColor, (int)(dx * 4u), (int)dy);
        surf2Dwrite((uint8_t)0u,              dstValid, (int)dx,         (int)dy);
    }
}

// ---------------------------------------------------------------------------
// Stage 3 — kernel_pull
//
// For each hole pixel at the current (finer) level, bilinear-samples the
// one-coarser level and writes the interpolated colour, marking the pixel
// as valid.  Valid pixels are left unchanged.
// ---------------------------------------------------------------------------

__global__ void kernel_pull(
    cudaSurfaceObject_t dstColor,
    cudaSurfaceObject_t dstValid,
    cudaTextureObject_t coarserTex,   // RGBA8, normalised float, bilinear
    uint32_t dstW, uint32_t dstH)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    uint8_t v;
    surf2Dread(&v, dstValid, (int)x, (int)y);
    if (v != 0u) return;   // valid pixel — no fill needed

    // Map pixel centre (x+0.5, y+0.5) at the current level into the
    // coarser level's unnormalised texture coordinate space (halved).
    const float cx = (x + 0.5f) * 0.5f;
    const float cy = (y + 0.5f) * 0.5f;
    const float4 f = tex2D<float4>((cudaTextureObject_t)coarserTex, cx, cy);

    const uchar4 filled = make_uchar4(
        (uint8_t)fminf(f.x * 255.0f + 0.5f, 255.0f),
        (uint8_t)fminf(f.y * 255.0f + 0.5f, 255.0f),
        (uint8_t)fminf(f.z * 255.0f + 0.5f, 255.0f),
        (uint8_t)fminf(f.w * 255.0f + 0.5f, 255.0f));

    surf2Dwrite(filled,        dstColor, (int)(x * 4u), (int)y);
    surf2Dwrite((uint8_t)255u, dstValid, (int)x,        (int)y);
}

// ---------------------------------------------------------------------------
// Stage 4 — kernel_writeback
//
// Writes filled colours from pyramid[0] back to the original frame, but
// only at pixels that were originally holes (holeMap == 255).
// Valid pixels in frame are never modified.
// ---------------------------------------------------------------------------

__global__ void kernel_writeback(
    cudaSurfaceObject_t frame,
    cudaSurfaceObject_t pyramid0Color,
    cudaSurfaceObject_t holeMap,
    uint32_t width, uint32_t height)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint8_t hole;
    surf2Dread(&hole, holeMap, (int)x, (int)y);
    if (hole == 0u) return;   // valid pixel — leave frame unchanged

    uchar4 filled;
    surf2Dread(&filled, pyramid0Color, (int)(x * 4u), (int)y);
    surf2Dwrite(filled, frame, (int)(x * 4u), (int)y);
}

// ---------------------------------------------------------------------------
// HoleFiller — constructor
// ---------------------------------------------------------------------------

HoleFiller::HoleFiller(CUcontext /*ctx*/, uint32_t width, uint32_t height)
{
    // Compute number of levels: enough to reduce the largest dimension to 1.
    uint32_t maxDim = (width > height) ? width : height;
    int levels = 0;
    for (uint32_t d = maxDim; d > 0; d >>= 1)
        ++levels;
    m_numLevels = (levels < 12) ? levels : 12;

    m_pyramid.resize(m_numLevels);

    uint32_t w = width, h = height;
    for (int l = 0; l < m_numLevels; ++l) {
        CUDA_ARRAY3D_DESCRIPTOR ad = {};
        ad.Width  = w;
        ad.Height = h;
        ad.Depth  = 0;
        ad.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        ad.Flags  = CUDA_ARRAY3D_SURFACE_LDST;

        ad.NumChannels = 4;
        CHECK_CU(cuArray3DCreate(&m_pyramid[l].color, &ad));

        ad.NumChannels = 1;
        CHECK_CU(cuArray3DCreate(&m_pyramid[l].valid, &ad));

        m_pyramid[l].w = w;
        m_pyramid[l].h = h;

        // Ceiling division: ensures every dimension reaches 1 eventually.
        w = std::max(1u, (w + 1u) / 2u);
        h = std::max(1u, (h + 1u) / 2u);
    }
}

// ---------------------------------------------------------------------------
// HoleFiller — destructor
// ---------------------------------------------------------------------------

HoleFiller::~HoleFiller()
{
    destroy();
}

void HoleFiller::destroy() noexcept
{
    for (auto& lv : m_pyramid) {
        if (lv.valid) { cuArrayDestroy(lv.valid); lv.valid = nullptr; }
        if (lv.color) { cuArrayDestroy(lv.color); lv.color = nullptr; }
    }
    m_pyramid.clear();
}

// ---------------------------------------------------------------------------
// HoleFiller — fill
// ---------------------------------------------------------------------------

void HoleFiller::fill(CUarray frame, CUarray holeMap, CUstream /*stream*/)
{
    const dim3 block(16, 16);

    // --- Stage 1: copy frame + holeMap into pyramid[0] ---
    {
        CUsurfObject surfFrame  = makeSurf(frame);
        CUsurfObject surfHole   = makeSurf(holeMap);
        CUsurfObject surfColor0 = makeSurf(m_pyramid[0].color);
        CUsurfObject surfValid0 = makeSurf(m_pyramid[0].valid);

        const dim3 grid((m_pyramid[0].w + 15) / 16,
                        (m_pyramid[0].h + 15) / 16);
        kernel_copy_level0<<<grid, block>>>(
            (cudaSurfaceObject_t)surfFrame,
            (cudaSurfaceObject_t)surfHole,
            (cudaSurfaceObject_t)surfColor0,
            (cudaSurfaceObject_t)surfValid0,
            m_pyramid[0].w, m_pyramid[0].h);

        cuSurfObjectDestroy(surfValid0);
        cuSurfObjectDestroy(surfColor0);
        cuSurfObjectDestroy(surfHole);
        cuSurfObjectDestroy(surfFrame);
    }

    // --- Stage 2: push (downsample from level 0 to numLevels-1) ---
    for (int l = 1; l < m_numLevels; ++l) {
        CUsurfObject srcColor = makeSurf(m_pyramid[l - 1].color);
        CUsurfObject srcValid = makeSurf(m_pyramid[l - 1].valid);
        CUsurfObject dstColor = makeSurf(m_pyramid[l].color);
        CUsurfObject dstValid = makeSurf(m_pyramid[l].valid);

        const dim3 grid((m_pyramid[l].w + 15) / 16,
                        (m_pyramid[l].h + 15) / 16);
        kernel_push<<<grid, block>>>(
            (cudaSurfaceObject_t)srcColor,
            (cudaSurfaceObject_t)srcValid,
            (cudaSurfaceObject_t)dstColor,
            (cudaSurfaceObject_t)dstValid,
            m_pyramid[l - 1].w, m_pyramid[l - 1].h,
            m_pyramid[l].w,     m_pyramid[l].h);

        cuSurfObjectDestroy(dstValid);
        cuSurfObjectDestroy(dstColor);
        cuSurfObjectDestroy(srcValid);
        cuSurfObjectDestroy(srcColor);
    }

    // --- Stage 3: pull (upsample from numLevels-2 down to 0) ---
    for (int l = m_numLevels - 2; l >= 0; --l) {
        CUtexObject  coarserTex = makeTexRGBA8(m_pyramid[l + 1].color);
        CUsurfObject dstColor   = makeSurf(m_pyramid[l].color);
        CUsurfObject dstValid   = makeSurf(m_pyramid[l].valid);

        const dim3 grid((m_pyramid[l].w + 15) / 16,
                        (m_pyramid[l].h + 15) / 16);
        kernel_pull<<<grid, block>>>(
            (cudaSurfaceObject_t)dstColor,
            (cudaSurfaceObject_t)dstValid,
            (cudaTextureObject_t)coarserTex,
            m_pyramid[l].w, m_pyramid[l].h);

        cuSurfObjectDestroy(dstValid);
        cuSurfObjectDestroy(dstColor);
        cuTexObjectDestroy(coarserTex);
    }

    // --- Stage 4: write filled pixels back to the original frame ---
    {
        CUsurfObject surfFrame  = makeSurf(frame);
        CUsurfObject surfColor0 = makeSurf(m_pyramid[0].color);
        CUsurfObject surfHole   = makeSurf(holeMap);

        const dim3 grid((m_pyramid[0].w + 15) / 16,
                        (m_pyramid[0].h + 15) / 16);
        kernel_writeback<<<grid, block>>>(
            (cudaSurfaceObject_t)surfFrame,
            (cudaSurfaceObject_t)surfColor0,
            (cudaSurfaceObject_t)surfHole,
            m_pyramid[0].w, m_pyramid[0].h);

        cuSurfObjectDestroy(surfHole);
        cuSurfObjectDestroy(surfColor0);
        cuSurfObjectDestroy(surfFrame);
    }

    // Synchronise (stream parameter reserved for future async integration)
    CHECK_CU(cuCtxSynchronize());
}
