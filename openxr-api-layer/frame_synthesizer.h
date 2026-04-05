#pragma once

#include <cuda.h>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

// ---------------------------------------------------------------------------
// Error-check macros (guarded so they can coexist with ofa_pipeline.h)
// ---------------------------------------------------------------------------

#ifndef CHECK_CU
#define CHECK_CU(call)                                                            \
    do {                                                                          \
        CUresult _r = (call);                                                     \
        if (_r != CUDA_SUCCESS) {                                                 \
            const char* _s = nullptr;                                             \
            cuGetErrorString(_r, &_s);                                            \
            throw std::runtime_error(std::string("CUDA driver: ") +              \
                                     (_s ? _s : "unknown error"));                \
        }                                                                         \
    } while (0)
#endif

// ---------------------------------------------------------------------------
// SynthFlowVector
//
// Motion vector in S10.5 fixed-point format.
// Binary-compatible with NV_OF_FLOW_VECTOR from nvOpticalFlowCommon.h.
// Divide by 32.0f to convert to pixel displacement.
//
// NOTE: OFAPipeline yields N→N-1 (backward) flow. Negate at the call site
// before passing to loadMotionVectors(), which expects N→N+1 (forward) flow.
// ---------------------------------------------------------------------------

struct SynthFlowVector {
    int16_t flowx;
    int16_t flowy;
};

// NV_OF_FLOW_VECTOR is { int16_t flowx; int16_t flowy; } — 4 bytes, same layout.
// These asserts verify binary compatibility without requiring NvOF headers here.
static_assert(sizeof(SynthFlowVector) == 4,
    "SynthFlowVector size mismatch with NV_OF_FLOW_VECTOR (expected 4 bytes)");
static_assert(offsetof(SynthFlowVector, flowx) == 0,
    "SynthFlowVector::flowx offset mismatch");
static_assert(offsetof(SynthFlowVector, flowy) == 2,
    "SynthFlowVector::flowy offset mismatch");
static_assert(std::is_trivially_copyable_v<SynthFlowVector>,
    "SynthFlowVector must be trivially copyable for memcpy from OFA output buffer");

// ---------------------------------------------------------------------------
// FrameSynthesizer
//
// Synthesizes an intermediate frame at T+0.5 between frame N (at T) and
// frame N+1 (at T+1), given a dense motion vector field and optional depth.
//
// Pipeline (two CUDA passes):
//   Pass 1 — Atomic scatter: threads from both source frames compete to write
//             a packed (depth, vector) value to each T+0.5 output pixel via
//             atomicMin. The closest-depth (smallest depth value) source wins.
//   Pass 2 — Bilinear gather+blend: each output pixel uses the winning vector
//             to bilinear-sample both frames and blends 50/50.
//
// The class does NOT own input CUarrays — callers must keep them alive until
// execute() returns. Output CUarrays (synthesizedFrame, holeMap) are owned
// by this object.
// ---------------------------------------------------------------------------

class FrameSynthesizer {
public:
    // ctx         : active CUDA context — must remain valid for lifetime of this object
    // width/height: frame dimensions in pixels
    // ofaGridSize : OFA output grid size (4 = one vector per 4×4 pixel block)
    FrameSynthesizer(CUcontext ctx, uint32_t width, uint32_t height, uint32_t ofaGridSize = 4);
    ~FrameSynthesizer();

    FrameSynthesizer(const FrameSynthesizer&)            = delete;
    FrameSynthesizer& operator=(const FrameSynthesizer&) = delete;
    FrameSynthesizer(FrameSynthesizer&&)                 = delete;
    FrameSynthesizer& operator=(FrameSynthesizer&&)      = delete;

    // Load the two input frames.
    //   frame : RGBA8 CUarray (from SharedImage::cuArray() in live use,
    //           or cudaMallocArray in offline tests)
    //   depth : R32F CUarray with per-pixel depth in [0,1], where smaller = closer
    //           to the camera. Pass nullptr to use a uniform default depth of 0.5
    //           for all pixels (correct for depth-unaware synthesis).
    // Not owned — caller is responsible for keeping alive until execute() returns.
    void loadFrameN (CUarray frame, CUarray depth = nullptr);
    void loadFrameN1(CUarray frame, CUarray depth = nullptr);

    // Upload motion vectors from the OFA grid.
    //   hostVectors : forward flow N→N+1 in S10.5 fixed-point
    //                 (binary-compatible with NV_OF_FLOW_VECTOR)
    //   count       : must equal (width/ofaGridSize) * (height/ofaGridSize)
    // S10.5 → float conversion (÷32.0f) is performed at upload time so that
    // Pass 1 threads read float2 directly from the ~4 MB grid buffer, exploiting
    // L2 cache reuse (16 threads share each grid cell at 4×4 grid size).
    void loadMotionVectors(const SynthFlowVector* hostVectors, size_t count);

    // Execute the synthesis pipeline synchronously.
    // stream is reserved for future async integration; currently ignored.
    void execute(CUstream stream = nullptr);

    // Output CUarrays — owned by this object, valid until the next execute()
    // call or destruction.
    CUarray synthesizedFrame() const { return m_outFrame;   }  // RGBA8
    CUarray holeMap()          const { return m_outHoleMap; }  // R8: 0=valid, 255=hole

private:
    void destroy() noexcept;

    uint32_t m_width{};
    uint32_t m_height{};
    uint32_t m_ofaGridSize{};
    uint32_t m_gridW{};    // width  / ofaGridSize
    uint32_t m_gridH{};    // height / ofaGridSize

    // ~4 MB at 4K — float2 per OFA grid cell, uploaded from host S10.5 input
    CUdeviceptr m_vecBufGrid{};

    // Full-res uint64_t atomic scatter target — one entry per output pixel
    CUdeviceptr m_scatterBuf{};

    // Owned output arrays
    CUarray m_outFrame{};      // RGBA8 synthesized output
    CUarray m_outHoleMap{};    // R8   hole map

    // Borrowed input arrays (not owned)
    CUarray m_frameN{};
    CUarray m_frameN1{};
    CUarray m_depthN{};
    CUarray m_depthN1{};
};
