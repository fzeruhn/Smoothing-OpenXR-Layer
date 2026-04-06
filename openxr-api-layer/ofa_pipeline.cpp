// openxr-api-layer/ofa_pipeline.cpp
// Note: uses NvOF and CUDA headers directly — must NOT use the project PCH.
//
// NvOFAPICreateInstanceCuda is NOT a static export — it lives in nvofapi64.dll
// which ships with the NVIDIA driver. We load it dynamically the same way the
// NvOF SDK samples do (see Common/NvOFBase/NvOF.cpp + NvOFCuda.cpp in the SDK).

#include "ofa_pipeline.h"

#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// ---------------------------------------------------------------------------
// loadNvOFAPI  — load nvofapi64.dll and return NvOFAPICreateInstanceCuda
// ---------------------------------------------------------------------------
namespace {

typedef NV_OF_STATUS(NVOFAPI *PFNNvOFAPICreateInstanceCuda)(
    uint32_t apiVer, NV_OF_CUDA_API_FUNCTION_LIST* functionList);

// Loads nvofapi64.dll, stores the module handle in *outModule, and returns the
// NvOFAPICreateInstanceCuda function pointer.  The caller is responsible for
// calling FreeLibrary(*outModule) when done.
PFNNvOFAPICreateInstanceCuda loadEntryPoint(void** outModule) {
#ifdef _WIN32
    HMODULE hMod = LoadLibraryA("nvofapi64.dll");
    if (!hMod)
        throw std::runtime_error(
            "Failed to load nvofapi64.dll — NVIDIA driver may not support OFA "
            "or the driver version is too old (Turing or later required).");
    auto fn = reinterpret_cast<PFNNvOFAPICreateInstanceCuda>(
        GetProcAddress(hMod, "NvOFAPICreateInstanceCuda"));
    if (!fn) {
        FreeLibrary(hMod);
        throw std::runtime_error(
            "NvOFAPICreateInstanceCuda not found in nvofapi64.dll — "
            "driver does not expose the OFA CUDA API.");
    }
    *outModule = static_cast<void*>(hMod);
    return fn;
#else
    (void)outModule;
    throw std::runtime_error("OFAPipeline: only Windows is supported.");
#endif
}

} // namespace

// -------------------------------------------------------------------------
// Constructor
// -------------------------------------------------------------------------
OFAPipeline::OFAPipeline(CUcontext ctx,
                         uint32_t  width,
                         uint32_t  height,
                         NV_OF_OUTPUT_VECTOR_GRID_SIZE gridSize,
                         NV_OF_PERF_LEVEL              perfLevel)
    : m_width(width)
    , m_height(height)
    , m_outW(width  / static_cast<uint32_t>(gridSize))
    , m_outH(height / static_cast<uint32_t>(gridSize))
    , m_hostOutput(m_outW * m_outH)
{
    // 1. Dynamically load NvOF function pointer table from the driver DLL.
    // NV_OF_CUDA_API_FUNCTION_LIST has no .size field — zero-init is correct.
    auto NvOFAPICreateInstanceCuda = loadEntryPoint(&m_hModule);

    // Guard: if anything below throws, the destructor won't run (constructor
    // didn't complete), so we must free the module handle ourselves.
    try {
        CHECK_NVOF_INIT(NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &m_api));

        // 2. Create OFA instance bound to CUDA context.
        CHECK_NVOF(m_api.nvCreateOpticalFlowCuda(ctx, &m_hOf));

        // 3. Configure and initialise OFA.
        NV_OF_INIT_PARAMS params{};
        params.width               = width;
        params.height              = height;
        params.outGridSize         = gridSize;
        params.hintGridSize        = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
        params.mode                = NV_OF_MODE_OPTICALFLOW;
        params.perfLevel           = perfLevel;
        params.enableExternalHints = NV_OF_FALSE;
        params.enableOutputCost    = NV_OF_FALSE;
        params.enableRoi           = NV_OF_FALSE;
        params.predDirection       = NV_OF_PRED_DIRECTION_FORWARD;
        params.enableGlobalFlow    = NV_OF_FALSE;
        params.inputBufferFormat   = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
        CHECK_NVOF(m_api.nvOFInit(m_hOf, &params));

        // 4. Create two input buffers (slot 0 = current, slot 1 = reference).
        NV_OF_BUFFER_DESCRIPTOR inputDesc{};
        inputDesc.width        = width;
        inputDesc.height       = height;
        inputDesc.bufferUsage  = NV_OF_BUFFER_USAGE_INPUT;
        inputDesc.bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
        for (int i = 0; i < 2; ++i)
            CHECK_NVOF(m_api.nvOFCreateGPUBufferCuda(m_hOf, &inputDesc,
                       NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_inputBufs[i]));

        // 5. Create output buffer (NV_OF_FLOW_VECTOR = SHORT2, one entry per grid cell).
        NV_OF_BUFFER_DESCRIPTOR outputDesc{};
        outputDesc.width        = m_outW;
        outputDesc.height       = m_outH;
        outputDesc.bufferUsage  = NV_OF_BUFFER_USAGE_OUTPUT;
        outputDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
        CHECK_NVOF(m_api.nvOFCreateGPUBufferCuda(m_hOf, &outputDesc,
                   NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_outputBuf));
    } catch (...) {
        // Constructor didn't complete — destructor won't run. Release any
        // OFA handles and the DLL that were acquired before the failure.
        // destroy() is safe to call with partial state (checks each handle).
        destroy();
        throw;
    }
}

// -------------------------------------------------------------------------
// Destructor
// -------------------------------------------------------------------------
OFAPipeline::~OFAPipeline() { destroy(); }

void OFAPipeline::destroy() noexcept {
    // Destroy GPU buffer handles before OFA instance (reverse init order).
    if (m_outputBuf)    { m_api.nvOFDestroyGPUBufferCuda(m_outputBuf);    m_outputBuf    = nullptr; }
    if (m_inputBufs[1]) { m_api.nvOFDestroyGPUBufferCuda(m_inputBufs[1]); m_inputBufs[1] = nullptr; }
    if (m_inputBufs[0]) { m_api.nvOFDestroyGPUBufferCuda(m_inputBufs[0]); m_inputBufs[0] = nullptr; }
    if (m_hOf)          { m_api.nvOFDestroy(m_hOf);                        m_hOf          = nullptr; }
    // Release the DLL reference. m_api function pointers become invalid after this.
#ifdef _WIN32
    if (m_hModule)      { FreeLibrary(static_cast<HMODULE>(m_hModule));    m_hModule      = nullptr; }
#endif
}

// -------------------------------------------------------------------------
// loadFrame
// -------------------------------------------------------------------------
void OFAPipeline::loadFrame(int slot, const void* hostGray8) {
    // Get stride info to respect OFA's pitch alignment.
    NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
    CHECK_NVOF(m_api.nvOFGPUBufferGetStrideInfo(m_inputBufs[slot], &si));

    CUdeviceptr devPtr = m_api.nvOFGPUBufferGetCUdeviceptr(m_inputBufs[slot]);

    CUDA_MEMCPY2D cp{};
    cp.srcMemoryType  = CU_MEMORYTYPE_HOST;
    cp.srcHost        = hostGray8;
    cp.srcPitch       = m_width;                          // tightly packed host data
    cp.dstMemoryType  = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice      = devPtr;
    cp.dstPitch       = si.strideInfo[0].strideXInBytes;  // OFA-allocated pitch
    cp.WidthInBytes   = m_width;                          // 1 byte per pixel (GRAYSCALE8)
    cp.Height         = m_height;
    CHECK_CU(cuMemcpy2D(&cp));
}

void OFAPipeline::loadFrameDevice(int slot, CUdeviceptr deviceGray8, size_t srcPitch, CUstream stream) {
    NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
    CHECK_NVOF(m_api.nvOFGPUBufferGetStrideInfo(m_inputBufs[slot], &si));

    CUdeviceptr devPtr = m_api.nvOFGPUBufferGetCUdeviceptr(m_inputBufs[slot]);

    CUDA_MEMCPY2D cp{};
    cp.srcMemoryType  = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice      = deviceGray8;
    cp.srcPitch       = srcPitch;
    cp.dstMemoryType  = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice      = devPtr;
    cp.dstPitch       = si.strideInfo[0].strideXInBytes;
    cp.WidthInBytes   = m_width;
    cp.Height         = m_height;

    if (stream != nullptr) {
        CHECK_CU(cuMemcpy2DAsync(&cp, stream));
    } else {
        CHECK_CU(cuMemcpy2D(&cp));
    }
}

// -------------------------------------------------------------------------
// execute
// -------------------------------------------------------------------------
void OFAPipeline::execute(CUstream /*stream*/, bool readbackToHost) {
    // `stream` is reserved for Item 4 pre-warp integration.
    // OFA uses its own internal stream; caller must cuCtxSynchronize() after.

    NV_OF_EXECUTE_INPUT_PARAMS in{};
    in.inputFrame           = m_inputBufs[0];  // current frame
    in.referenceFrame       = m_inputBufs[1];  // previous frame
    in.disableTemporalHints = NV_OF_TRUE;       // no inter-frame predictor

    NV_OF_EXECUTE_OUTPUT_PARAMS out{};
    out.outputBuffer = m_outputBuf;

    CHECK_NVOF(m_api.nvOFExecute(m_hOf, &in, &out));

    if (readbackToHost) {
        // Copy motion vectors to host staging buffer (caller syncs before reading).
        NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
        CHECK_NVOF(m_api.nvOFGPUBufferGetStrideInfo(m_outputBuf, &si));

        CUdeviceptr outDev  = m_api.nvOFGPUBufferGetCUdeviceptr(m_outputBuf);
        const uint32_t rowBytes = m_outW * static_cast<uint32_t>(sizeof(NV_OF_FLOW_VECTOR));

        CUDA_MEMCPY2D cp{};
        cp.srcMemoryType  = CU_MEMORYTYPE_DEVICE;
        cp.srcDevice      = outDev;
        cp.srcPitch       = si.strideInfo[0].strideXInBytes;
        cp.dstMemoryType  = CU_MEMORYTYPE_HOST;
        cp.dstHost        = m_hostOutput.data();
        cp.dstPitch       = rowBytes;
        cp.WidthInBytes   = rowBytes;
        cp.Height         = m_outH;
        CHECK_CU(cuMemcpy2D(&cp));
    }
}

CUdeviceptr OFAPipeline::outputDevicePtr() const {
    if (!m_outputBuf) {
        return 0;
    }
    return m_api.nvOFGPUBufferGetCUdeviceptr(m_outputBuf);
}
