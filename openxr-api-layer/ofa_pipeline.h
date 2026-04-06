#pragma once

// NvOF C API — do not include through pch.h
#include <cuda.h>
#if __has_include("nvOpticalFlowCuda.h") && __has_include("nvOpticalFlowCommon.h")
#include "nvOpticalFlowCuda.h"
#include "nvOpticalFlowCommon.h"
#elif __has_include("../Optical_Flow_SDK_5.0.7/NvOFInterface/nvOpticalFlowCuda.h") && __has_include("../Optical_Flow_SDK_5.0.7/NvOFInterface/nvOpticalFlowCommon.h")
#include "../Optical_Flow_SDK_5.0.7/NvOFInterface/nvOpticalFlowCuda.h"
#include "../Optical_Flow_SDK_5.0.7/NvOFInterface/nvOpticalFlowCommon.h"
#elif __has_include("../../Optical_Flow_SDK_5.0.7/NvOFInterface/nvOpticalFlowCuda.h") && __has_include("../../Optical_Flow_SDK_5.0.7/NvOFInterface/nvOpticalFlowCommon.h")
#include "../../Optical_Flow_SDK_5.0.7/NvOFInterface/nvOpticalFlowCuda.h"
#include "../../Optical_Flow_SDK_5.0.7/NvOFInterface/nvOpticalFlowCommon.h"
#else
#error "NvOF headers not found. Add NvOFInterface to include paths or place Optical_Flow_SDK_5.0.7 next to the repo."
#endif

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error-check macros
// ---------------------------------------------------------------------------

// For the one call that happens before m_hOf exists (NvOFAPICreateInstanceCuda).
#define CHECK_NVOF_INIT(call)                                                     \
    do {                                                                          \
        NV_OF_STATUS _s = (call);                                                 \
        if (_s != NV_OF_SUCCESS)                                                  \
            throw std::runtime_error("NvOFAPICreateInstanceCuda failed: status " +\
                                     std::to_string(static_cast<int>(_s)));       \
    } while (0)

// For all subsequent NvOf calls (m_hOf is valid, error string available).
#define CHECK_NVOF(call)                                                          \
    do {                                                                          \
        NV_OF_STATUS _s = (call);                                                 \
        if (_s != NV_OF_SUCCESS) {                                                \
            char _buf[512] = {};                                                  \
            uint32_t _sz = sizeof(_buf);                                          \
            if (m_hOf) m_api.nvOFGetLastError(m_hOf, _buf, &_sz);                \
            throw std::runtime_error(std::string("NvOF error ") +                 \
                                     std::to_string(static_cast<int>(_s)) +       \
                                     ": " + _buf);                                \
        }                                                                         \
    } while (0)

// For CUDA driver API calls.
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

// ---------------------------------------------------------------------------
// OFAPipeline
// ---------------------------------------------------------------------------

class OFAPipeline {
public:
    // Construct and fully initialise the OFA instance.
    // ctx         : active CUDA context (must remain valid for the lifetime of this object)
    // width/height: input frame dimensions in pixels
    // gridSize    : output grid — NV_OF_OUTPUT_VECTOR_GRID_SIZE_4 gives one vector per 4x4 block
    // perfLevel   : quality/speed trade-off
    OFAPipeline(CUcontext ctx,
                uint32_t  width,
                uint32_t  height,
                NV_OF_OUTPUT_VECTOR_GRID_SIZE gridSize  = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4,
                NV_OF_PERF_LEVEL              perfLevel = NV_OF_PERF_LEVEL_MEDIUM);

    ~OFAPipeline();

    OFAPipeline(const OFAPipeline&)            = delete;
    OFAPipeline& operator=(const OFAPipeline&) = delete;
    OFAPipeline(OFAPipeline&&)                 = delete;
    OFAPipeline& operator=(OFAPipeline&&)      = delete;

    // Upload an 8-bit grayscale host frame into the named slot.
    //   slot 0 = inputFrame     (current / new frame)
    //   slot 1 = referenceFrame (previous frame)
    //
    // TODO (Item 4 pre-warp hook): replace or augment with
    //   void setInputDevicePtr(int slot, CUdeviceptr devPtr);
    // so the pre-warp kernel can write directly into OFA input buffers.
    void loadFrame(int slot, const void* hostGray8);
    void loadFrameDevice(int slot, CUdeviceptr deviceGray8, size_t srcPitch, CUstream stream = nullptr);

    // Submit OFA work.
    // `stream` is reserved for future integration; currently ignored because NvOF
    // manages execution internally.
    // If readbackToHost=false, skips host readback so hot paths can remain GPU-only.
    void execute(CUstream stream = nullptr, bool readbackToHost = true);

    // Pointer into the host-side readback buffer.
    // Valid only when execute(..., readbackToHost=true) has been used and
    // the work is synchronized. If readbackToHost=false, this buffer is not
    // refreshed; use outputDevicePtr() for GPU consumers.
    // Each element is NV_OF_FLOW_VECTOR { int16_t flowx, flowy } in S10.5 fixed-point.
    // Divide by 32.0f to convert to pixel displacement.
    const NV_OF_FLOW_VECTOR* outputData() const { return m_hostOutput.data(); }

    // Device pointer to OFA output vectors (NV_OF_FLOW_VECTOR / SHORT2 layout).
    CUdeviceptr outputDevicePtr() const;

    uint32_t outputWidth()  const { return m_outW; }  // width  / gridSize
    uint32_t outputHeight() const { return m_outH; }  // height / gridSize

private:
    void destroy() noexcept;

    void*                        m_hModule{nullptr};  // HMODULE for nvofapi64.dll (opaque to avoid <windows.h> in header)
    NV_OF_CUDA_API_FUNCTION_LIST m_api{};
    NvOFHandle                   m_hOf{};
    NvOFGPUBufferHandle          m_inputBufs[2]{};
    NvOFGPUBufferHandle          m_outputBuf{};

    uint32_t m_width{};
    uint32_t m_height{};
    uint32_t m_outW{};
    uint32_t m_outH{};

    std::vector<NV_OF_FLOW_VECTOR> m_hostOutput;
};
