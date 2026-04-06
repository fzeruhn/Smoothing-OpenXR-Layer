#pragma once

#include <cuda.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error-check macro (guarded so it can coexist with frame_synthesizer.h)
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
// HoleFiller
//
// Fills disocclusion holes in a synthesized frame using a hierarchical
// push-pull algorithm:
//
//   Push (downsample): Build a mipmap pyramid, averaging only valid pixels
//     at each level.  By the coarsest level, holes are filled by
//     contributions from distant valid neighbours.
//
//   Pull (upsample): Traverse back up the pyramid.  At each level, bilinear-
//     sample the one-coarser level to fill hole pixels, then leave valid
//     pixels unchanged.
//
// Interface is intentionally stable to allow future swap-in of an AI
// inpainting model (small U-Net, diffusion) without modifying callers.
//
// Requirements:
//   frame and holeMap must be allocated with CUDA_ARRAY3D_SURFACE_LDST.
//   FrameSynthesizer output arrays already satisfy this.
// ---------------------------------------------------------------------------

class HoleFiller {
public:
    // Allocates push-pull pyramid storage for frames of this size.
    // ctx must remain valid for the lifetime of this object.
    HoleFiller(CUcontext ctx, uint32_t width, uint32_t height);
    ~HoleFiller();

    HoleFiller(const HoleFiller&)            = delete;
    HoleFiller& operator=(const HoleFiller&) = delete;
    HoleFiller(HoleFiller&&)                 = delete;
    HoleFiller& operator=(HoleFiller&&)      = delete;

    // Fills holes in `frame` in-place using push-pull propagation.
    //   frame:   RGBA8 CUarray — synthesized frame (hole pixels replaced in-place)
    //   holeMap: R8 CUarray   — 0=valid, 255=hole (read-only; not modified)
    //   stream:  stream used for async dispatch
    //   synchronize: if true, waits for completion before returning
    //
    // On return, every hole pixel in frame holds a smoothly interpolated color
    // derived from nearby valid pixels.
    void fill(CUarray frame, CUarray holeMap, CUstream stream = nullptr, bool synchronize = true);

private:
    void destroy() noexcept;

    struct Level {
        CUarray  color{};   // RGBA8: colours at this pyramid level (holes = black)
        CUarray  valid{};   // R8: 255=valid pixel, 0=hole at this level
        uint32_t w{};
        uint32_t h{};
    };

    // m_pyramid[0]           = full-resolution workspace copy of the input frame.
    // m_pyramid[1..numLevels-1] = successively halved levels down to ~1×1.
    std::vector<Level> m_pyramid;
    int m_numLevels{};
};
