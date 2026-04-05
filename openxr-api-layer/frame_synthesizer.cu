// frame_synthesizer.cu
// Bidirectional frame synthesis — CUDA kernels and FrameSynthesizer implementation.
//
// Two-pass pipeline:
//   Pass 1 (kernel_scatter): For each source pixel in frame N (forward) and
//     frame N+1 (backward), atomically write a packed (depth, vector) value to
//     the T+0.5 destination using atomicMin. The closest-depth source wins.
//
//   Pass 2 (kernel_gather_blend): For each output pixel, unpack the winning
//     vector from the scatter buffer, bilinear-sample both input frames, and
//     blend 50/50. Pixels with no scatter coverage are marked as holes.
//
// Vector packing in the uint64 scatter buffer:
//   bits [63:32] = depth as raw IEEE 754 float bits (smaller float = closer = wins)
//   bits [31:16] = vx quantized to int16 (×128 precision, ~0.008 px/unit)
//   bits [15: 0] = vy quantized to int16 (×128 precision)
//
// Texture coordinate convention: CUDA unnormalised coords place pixel (i,j)
// at (i+0.5, j+0.5). All tex2D calls add 0.5 to match this.

#include "frame_synthesizer.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Runtime-API error check (local to this translation unit)
// ---------------------------------------------------------------------------

#define CHECK_RT(call)                                                            \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess)                                                    \
            throw std::runtime_error(std::string("CUDA runtime: ") +             \
                                     cudaGetErrorString(_e));                     \
    } while (0)

// ---------------------------------------------------------------------------
// File-local helpers — texture and surface object creation
// ---------------------------------------------------------------------------

// Create a bilinear-sampled texture object from an RGBA8 or R32F CUarray.
// normalizedFloat=true  → uint8 input is mapped to [0.0, 1.0] float output.
// normalizedFloat=false → element type is returned as-is (use for R32F depth).
static CUtexObject makeTexObject(CUarray arr, bool normalizedFloat)
{
    CUDA_RESOURCE_DESC rd = {};
    rd.resType             = CU_RESOURCE_TYPE_ARRAY;
    rd.res.array.hArray    = arr;

    CUDA_TEXTURE_DESC td = {};
    td.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    td.filterMode     = CU_TR_FILTER_MODE_LINEAR;
    // For RGBA8 (normalizedFloat=true): flags=0 keeps the default behaviour where
    // uint8 channel values are promoted to float [0,1] by the texture hardware.
    // For R32F depth (normalizedFloat=false): flags=0 as well — CU_TRSF_READ_AS_INTEGER
    // is only defined for integer-type arrays and must not be set on float arrays.
    // Filtering mode is controlled by td.filterMode above, not by this flag.
    td.flags = 0u;

    CUtexObject tex = 0;
    CHECK_CU(cuTexObjectCreate(&tex, &rd, &td, nullptr));
    return tex;
}

static CUsurfObject makeSurfObject(CUarray arr)
{
    CUDA_RESOURCE_DESC rd = {};
    rd.resType          = CU_RESOURCE_TYPE_ARRAY;
    rd.res.array.hArray = arr;

    CUsurfObject surf = 0;
    CHECK_CU(cuSurfObjectCreate(&surf, &rd));
    return surf;
}

// ---------------------------------------------------------------------------
// Pass 1 — kernel_scatter
//
// Launches once for frame N (isFrameN1=false, forward scatter) and once for
// frame N+1 (isFrameN1=true, backward scatter).  Each thread handles one
// source pixel and competes atomically for its T+0.5 destination.
// ---------------------------------------------------------------------------

__global__ void kernel_scatter(
    unsigned long long* __restrict__ scatterBuf,
    const float2*       __restrict__ vecGrid,
    cudaTextureObject_t              depthTex,   // R32F; 0 if depth unavailable
    bool                             hasDepth,
    uint32_t srcW, uint32_t srcH,
    uint32_t dstW, uint32_t dstH,
    uint32_t gridW, uint32_t gridSize,
    bool     isFrameN1)
{
    const uint32_t sx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t sy = blockIdx.y * blockDim.y + threadIdx.y;
    if (sx >= srcW || sy >= srcH) return;

    // Each thread reads one grid cell; 16 threads share each cell at gridSize=4.
    // The ~4 MB grid fits in L2 — standard global reads suffice.
    const float2 v = vecGrid[(sy / gridSize) * gridW + (sx / gridSize)];

    // Depth for this source pixel: smaller value = closer to camera.
    float depth;
    if (hasDepth) {
        // tex2D with CU_TRSF_READ_AS_INTEGER=1 → raw float element
        depth = tex2D<float>((cudaTextureObject_t)depthTex,
                             (float)sx + 0.5f, (float)sy + 0.5f);
    } else {
        depth = 0.5f;
    }

    // Compute T+0.5 destination.
    // Forward  (isFrameN1=false): N pixel at (sx,sy) moves toward N+1 by V.
    //   T+0.5 destination = (sx + 0.5*vx, sy + 0.5*vy)
    // Backward (isFrameN1=true):  N+1 pixel at (sx,sy) came from N at (sx-vx, sy-vy).
    //   T+0.5 destination = (sx - 0.5*vx, sy - 0.5*vy)
    float tx, ty;
    if (!isFrameN1) {
        tx = (float)sx + 0.5f * v.x;
        ty = (float)sy + 0.5f * v.y;
    } else {
        tx = (float)sx - 0.5f * v.x;
        ty = (float)sy - 0.5f * v.y;
    }

    const int ox = __float2int_rn(tx);
    const int oy = __float2int_rn(ty);
    if (ox < 0 || oy < 0 || (uint32_t)ox >= dstW || (uint32_t)oy >= dstH) return;

    // Pack (depth, vx, vy) into uint64.
    // IEEE 754 float bits in [0,1] preserve ordering when compared as uint32,
    // so atomicMin on the full uint64 naturally selects the closest pixel.
    const uint32_t depthBits = __float_as_uint(depth);

    // Quantise vector to int16 at ×128 sub-pixel precision (~0.008 px/unit).
    // Clamp to int16 range to handle large displacements safely.
    int vxQ = __float2int_rn(v.x * 128.0f);
    int vyQ = __float2int_rn(v.y * 128.0f);
    vxQ = max(-32767, min(32767, vxQ));
    vyQ = max(-32767, min(32767, vyQ));
    const uint32_t vecBits =
        ((uint32_t)(uint16_t)(int16_t)vxQ << 16) |
         (uint32_t)(uint16_t)(int16_t)vyQ;

    const unsigned long long packed =
        ((unsigned long long)depthBits << 32) | (unsigned long long)vecBits;

    atomicMin(&scatterBuf[(uint32_t)oy * dstW + (uint32_t)ox], packed);
}

// ---------------------------------------------------------------------------
// Pass 2 — kernel_gather_blend
//
// For each output pixel, unpacks the winning vector from the scatter buffer,
// bilinear-samples both input frames, and blends 50/50.
// Pixels with no scatter coverage (sentinel value) are marked as holes.
// ---------------------------------------------------------------------------

__global__ void kernel_gather_blend(
    cudaSurfaceObject_t              outFrame,
    cudaSurfaceObject_t              outHoleMap,
    const unsigned long long* __restrict__ scatterBuf,
    cudaTextureObject_t              texN,
    cudaTextureObject_t              texN1,
    uint32_t width, uint32_t height)
{
    const uint32_t ox = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= width || oy >= height) return;

    const unsigned long long val = scatterBuf[oy * width + ox];

    if (val == 0xFFFFFFFFFFFFFFFFULL) {
        // No scatter coverage — mark as hole.
        surf2Dwrite(make_uchar4(0, 0, 0, 0), outFrame,   (int)(ox * 4), (int)oy);
        surf2Dwrite((uint8_t)255,             outHoleMap, (int)ox,       (int)oy);
        return;
    }

    // Unpack vector (int16 ×128 fixed-point → float pixels).
    const uint32_t vecBits = (uint32_t)(val & 0xFFFFFFFFULL);
    const float vx = (float)(int16_t)((vecBits >> 16) & 0xFFFF) / 128.0f;
    const float vy = (float)(int16_t)( vecBits         & 0xFFFF) / 128.0f;

    // Sample frame N  at (ox - 0.5*vx, oy - 0.5*vy).
    // Sample frame N+1 at (ox + 0.5*vx, oy + 0.5*vy).
    // Add 0.5 to each coordinate to address the centre of the target pixel
    // in CUDA's unnormalised texture coordinate convention.
    const float sxN  = (float)ox - 0.5f * vx + 0.5f;
    const float syN  = (float)oy - 0.5f * vy + 0.5f;
    const float sxN1 = (float)ox + 0.5f * vx + 0.5f;
    const float syN1 = (float)oy + 0.5f * vy + 0.5f;

    // tex2D with normalised-float mode returns RGBA in [0.0, 1.0].
    const float4 cN  = tex2D<float4>(texN,  sxN,  syN);
    const float4 cN1 = tex2D<float4>(texN1, sxN1, syN1);

    // Equal-weight blend (depth-guided blend deferred to post-item-5).
    const float4 blended = make_float4(
        0.5f * cN.x + 0.5f * cN1.x,
        0.5f * cN.y + 0.5f * cN1.y,
        0.5f * cN.z + 0.5f * cN1.z,
        0.5f * cN.w + 0.5f * cN1.w);

    const uchar4 out = make_uchar4(
        (uint8_t)fminf(blended.x * 255.0f + 0.5f, 255.0f),
        (uint8_t)fminf(blended.y * 255.0f + 0.5f, 255.0f),
        (uint8_t)fminf(blended.z * 255.0f + 0.5f, 255.0f),
        (uint8_t)fminf(blended.w * 255.0f + 0.5f, 255.0f));

    surf2Dwrite(out,       outFrame,   (int)(ox * 4), (int)oy);
    surf2Dwrite((uint8_t)0, outHoleMap, (int)ox,       (int)oy);
}

// ---------------------------------------------------------------------------
// FrameSynthesizer — constructor
// ---------------------------------------------------------------------------

FrameSynthesizer::FrameSynthesizer(CUcontext /*ctx*/,
                                   uint32_t width, uint32_t height,
                                   uint32_t ofaGridSize)
    : m_width(width), m_height(height), m_ofaGridSize(ofaGridSize)
{
    m_gridW = (width  + ofaGridSize - 1) / ofaGridSize;
    m_gridH = (height + ofaGridSize - 1) / ofaGridSize;

    // float2 grid buffer (~4 MB at 4K / gridSize=4)
    CHECK_CU(cuMemAlloc(&m_vecBufGrid,
                        (size_t)m_gridW * m_gridH * sizeof(float2)));

    // uint64_t full-resolution scatter buffer
    CHECK_CU(cuMemAlloc(&m_scatterBuf,
                        (size_t)m_width * m_height * sizeof(uint64_t)));

    // RGBA8 output frame (needs surface-write support)
    CUDA_ARRAY3D_DESCRIPTOR ad = {};
    ad.Width       = m_width;
    ad.Height      = m_height;
    ad.Depth       = 0;
    ad.Format      = CU_AD_FORMAT_UNSIGNED_INT8;
    ad.NumChannels = 4;
    ad.Flags       = CUDA_ARRAY3D_SURFACE_LDST;
    CHECK_CU(cuArray3DCreate(&m_outFrame, &ad));

    // R8 hole map
    ad.NumChannels = 1;
    CHECK_CU(cuArray3DCreate(&m_outHoleMap, &ad));
}

// ---------------------------------------------------------------------------
// FrameSynthesizer — destructor
// ---------------------------------------------------------------------------

FrameSynthesizer::~FrameSynthesizer()
{
    destroy();
}

void FrameSynthesizer::destroy() noexcept
{
    if (m_outHoleMap) { cuArrayDestroy(m_outHoleMap); m_outHoleMap = nullptr; }
    if (m_outFrame)   { cuArrayDestroy(m_outFrame);   m_outFrame   = nullptr; }
    if (m_scatterBuf) { cuMemFree(m_scatterBuf);      m_scatterBuf = 0; }
    if (m_vecBufGrid) { cuMemFree(m_vecBufGrid);      m_vecBufGrid = 0; }
}

// ---------------------------------------------------------------------------
// FrameSynthesizer — input loading
// ---------------------------------------------------------------------------

void FrameSynthesizer::loadFrameN(CUarray frame, CUarray depth)
{
    m_frameN = frame;
    m_depthN = depth;
}

void FrameSynthesizer::loadFrameN1(CUarray frame, CUarray depth)
{
    m_frameN1 = frame;
    m_depthN1 = depth;
}

void FrameSynthesizer::loadMotionVectors(const SynthFlowVector* hostVectors,
                                          size_t count)
{
    // Convert S10.5 fixed-point → float2 on the host, then upload.
    // The ~4 MB grid buffer is read by 16 threads per cell in Pass 1;
    // L2 cache handles the reuse with zero additional DRAM traffic.
    std::vector<float2> converted(count);
    for (size_t i = 0; i < count; ++i) {
        converted[i].x = hostVectors[i].flowx / 32.0f;
        converted[i].y = hostVectors[i].flowy / 32.0f;
    }
    CHECK_CU(cuMemcpyHtoD(m_vecBufGrid,
                          converted.data(),
                          count * sizeof(float2)));
}

// ---------------------------------------------------------------------------
// FrameSynthesizer — execute
// ---------------------------------------------------------------------------

void FrameSynthesizer::execute(CUstream /*stream*/)
{
    // --- Initialise scatter buffer to sentinel (all bytes 0xFF) ---
    CHECK_CU(cuMemsetD8(m_scatterBuf, 0xFF,
                        (size_t)m_width * m_height * sizeof(uint64_t)));

    // --- Create texture objects for input frames ---
    CUtexObject texN  = makeTexObject(m_frameN,  true);   // RGBA8 normalised float
    CUtexObject texN1 = makeTexObject(m_frameN1, true);
    CUtexObject depthTexN  = m_depthN  ? makeTexObject(m_depthN,  false) : 0;
    CUtexObject depthTexN1 = m_depthN1 ? makeTexObject(m_depthN1, false) : 0;

    // --- Create surface objects for outputs ---
    CUsurfObject surfOut  = makeSurfObject(m_outFrame);
    CUsurfObject surfHole = makeSurfObject(m_outHoleMap);

    // 16×16 thread blocks — each thread handles one source pixel
    const dim3 block(16, 16);
    const dim3 grid((m_width  + 15) / 16,
                    (m_height + 15) / 16);

    // --- Pass 1a: forward scatter from frame N ---
    kernel_scatter<<<grid, block>>>(
        reinterpret_cast<unsigned long long*>(m_scatterBuf),
        reinterpret_cast<const float2*>(m_vecBufGrid),
        (cudaTextureObject_t)depthTexN,
        m_depthN != nullptr,
        m_width, m_height,
        m_width, m_height,
        m_gridW, m_ofaGridSize,
        false);

    // --- Pass 1b: backward scatter from frame N+1 ---
    kernel_scatter<<<grid, block>>>(
        reinterpret_cast<unsigned long long*>(m_scatterBuf),
        reinterpret_cast<const float2*>(m_vecBufGrid),
        (cudaTextureObject_t)depthTexN1,
        m_depthN1 != nullptr,
        m_width, m_height,
        m_width, m_height,
        m_gridW, m_ofaGridSize,
        true);

    // --- Pass 2: gather + blend ---
    kernel_gather_blend<<<grid, block>>>(
        (cudaSurfaceObject_t)surfOut,
        (cudaSurfaceObject_t)surfHole,
        reinterpret_cast<const unsigned long long*>(m_scatterBuf),
        (cudaTextureObject_t)texN,
        (cudaTextureObject_t)texN1,
        m_width, m_height);

    // Synchronise (stream parameter reserved for future async integration)
    CHECK_CU(cuCtxSynchronize());

    // --- Destroy temporary objects ---
    cuSurfObjectDestroy(surfHole);
    cuSurfObjectDestroy(surfOut);
    if (depthTexN1) cuTexObjectDestroy(depthTexN1);
    if (depthTexN)  cuTexObjectDestroy(depthTexN);
    cuTexObjectDestroy(texN1);
    cuTexObjectDestroy(texN);
}
