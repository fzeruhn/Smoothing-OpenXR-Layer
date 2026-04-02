# OFA Integration — Design Spec (Roadmap Item 3)

_Date: 2026-04-01_

---

## Context

Roadmap Item 3 wires up the NVIDIA Optical Flow Accelerator (NVOf SDK 5.0.7) as a standalone, validated subsystem. It sits between the completed Vulkan/CUDA interop foundation (Item 1) and the pre-warp stage (Item 4). The goal is a clean `OFAPipeline` class that accepts raw grayscale frames, runs OFA, and returns a dense motion vector field — validated in isolation before being wired into the real frame pipeline.

The real pipeline will feed `SharedImage`-backed CUDA surfaces into this class. That wiring is explicitly out of scope for this item; a hook slot is reserved for it (Item 4 pre-warp).

---

## New Files

```
openxr-api-layer/
  ofa_pipeline.h          OFAPipeline class declaration
  ofa_pipeline.cpp        OFAPipeline implementation
ofa-test/
  main.cpp                Standalone test harness
  stb_image_write.h       Header-only PNG writer (drop-in, no build system changes)
  ofa-test.vcxproj
```

---

## `OFAPipeline` Class (`ofa_pipeline.h/.cpp`)

### Interface

```cpp
class OFAPipeline {
public:
    OFAPipeline(CUcontext ctx,
                uint32_t width, uint32_t height,
                NV_OF_OUTPUT_VECTOR_GRID_SIZE gridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4,
                NV_OF_PERF_LEVEL perfLevel = NV_OF_PERF_LEVEL_MEDIUM);
    ~OFAPipeline();

    OFAPipeline(const OFAPipeline&) = delete;
    OFAPipeline& operator=(const OFAPipeline&) = delete;

    // Upload an 8-bit grayscale host frame into slot 0 (current) or 1 (reference).
    // TODO (Item 4): replace with a setPreWarpCallback() that receives
    //   CUdeviceptr current/reference so the pre-warp kernel runs before execute().
    void loadFrame(int slot, const void* hostGray8);

    // Submit OFA work asynchronously on `stream`. Caller must sync stream before
    // calling outputData(). Passing nullptr uses the null stream.
    void execute(CUstream stream = nullptr);

    // Pointer into host-side readback buffer. Valid after stream sync post-execute().
    const NV_OF_FLOW_VECTOR* outputData() const;

    uint32_t outputWidth()  const;  // == width  / gridSize
    uint32_t outputHeight() const;  // == height / gridSize
};
```

### Internals

| Member | Type | Purpose |
|---|---|---|
| `m_api` | `NV_OF_CUDA_API_FUNCTION_LIST` | Function pointers from `NvOFAPICreateInstanceCuda` |
| `m_hOf` | `NvOFHandle` | OFA instance handle |
| `m_inputBufs[2]` | `NvOFGPUBufferHandle` | `CUDEVICEPTR`, `GRAYSCALE8`, one per slot |
| `m_outputBuf` | `NvOFGPUBufferHandle` | `CUDEVICEPTR`, `NV_OF_FLOW_VECTOR` output |
| `m_hostOutput` | `std::vector<NV_OF_FLOW_VECTOR>` | CPU-side readback staging |
| `m_outW`, `m_outH` | `uint32_t` | `width/gridSize`, `height/gridSize` |

### Initialization sequence

1. `NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &m_api)` — loads all function pointers
2. `m_api.nvCreateOpticalFlowCuda(ctx, &m_hOf)` — create instance
3. `NV_OF_INIT_PARAMS` — set width/height, gridSize, perfLevel, `GRAYSCALE8`, `OPTICALFLOW`, forward only, no hints, no cost, no ROI
4. `m_api.nvOFInit(m_hOf, &initParams)`
5. `m_api.nvOFCreateGPUBufferCuda(...)` × 3 — two input buffers, one output buffer
6. Allocate `m_hostOutput` (`outW * outH` elements)

### Destruction order

1. Destroy three GPU buffer handles via `nvOFDestroyGPUBufferCuda`
2. Destroy instance via `nvOFDestroy`
3. `m_hostOutput` freed by vector destructor

### `execute()` sequence

```
NV_OF_EXECUTE_INPUT_PARAMS  in  = { m_inputBufs[0], m_inputBufs[1] };
NV_OF_EXECUTE_OUTPUT_PARAMS out = { m_outputBuf };
m_api.nvOFExecute(m_hOf, &in, &out);
// async copy output to m_hostOutput (cuMemcpy from CUDEVICEPTR)
```

---

## Test Harness (`ofa-test/main.cpp`)

### Synthetic input

| | Frame 0 (reference) | Frame 1 (current) |
|---|---|---|
| Content | Checkerboard (32px squares, 0/200 luma) | Frame 0 shifted +8px right, +4px down |
| Size | 256 × 256 grayscale | 256 × 256 grayscale |
| Generation | CPU, `std::vector<uint8_t>` | CPU, same |

### OFA configuration

- Grid size: `NV_OF_OUTPUT_VECTOR_GRID_SIZE_4` → output is 64 × 64
- Perf level: `NV_OF_PERF_LEVEL_MEDIUM`
- Input format: `GRAYSCALE8`

### Validation

Expected vector: `flowx = +256` S10.5 (= +8.0 px), `flowy = +128` S10.5 (= +4.0 px).

Pass criterion: ≥ 95% of vectors in the central 50 × 50 region satisfy:
```
|vec.flowx - 256| <= 1  &&  |vec.flowy - 128| <= 1
```
Border region (outer 7 cells) excluded due to edge artifacts.

### PNG output

Written to `ofa-test-output.png` via `stb_image_write`. Encoding:
- Map `flowx` → red channel: `clamp((flowx/32.0f + 16) / 32 * 255, 0, 255)`
- Map `flowy` → green channel: same formula
- Blue = 0

Simple false-color is sufficient for visual sanity-check.

### Console output

```
[PASS] OFA verified: 3187/2500 central vectors within tolerance
ofa-test-output.png written
```
(or `[FAIL]` with the first out-of-tolerance vector coordinates)

---

## Build System

### `openxr-api-layer.vcxproj` — modified

Add to `AdditionalIncludeDirectories` in x64 Debug and Release configs:
```
$(SolutionDir)..\Optical_Flow_SDK_5.0.7\NvOFInterface
```
No new lib link. No DLL deployment (NvOf is driver-resident).

### `ofa-test.vcxproj` — new

| Setting | Value |
|---|---|
| CUDA Build Customizations | `CUDA 13.2.props` |
| Additional Includes | `$(CudaToolkitDir)include`, `$(SolutionDir)..\Optical_Flow_SDK_5.0.7\NvOFInterface`, `$(SolutionDir)openxr-api-layer` |
| Additional Libs | `cuda.lib`, `cudart.lib` |
| Additional Lib Dirs | `$(CudaToolkitDir)lib\x64` |
| Output | `$(SolutionDir)bin\$(Platform)\$(Configuration)\ofa-test.exe` |

`stb_image_write.h` lives in `ofa-test/` — header-only, no vcxproj entry needed.

### `SMOOTHING-OPENXR-LAYER.sln` — modified

Add `ofa-test` project (same pattern as `interop-test` entry).

---

## Error Handling

All NvOf C API calls checked with a macro:
```cpp
#define CHECK_NVOF(call) \
    do { if ((call) != NV_OF_SUCCESS) { \
        char buf[512]; uint32_t sz = sizeof(buf); \
        m_api.nvOFGetLastError(m_hOf, buf, &sz); \
        throw std::runtime_error(std::string("NvOF: ") + buf); \
    } } while(0)
```

---

## Pre-Warp Hook (Item 4 reservation)

`loadFrame(slot, hostPtr)` is intentionally a simple `UploadData` call. When Item 4 is implemented, it will be replaced by:
```cpp
// Item 4: caller provides pre-warped CUdeviceptr directly,
// bypassing the host upload path.
void setInputDevicePtr(int slot, CUdeviceptr devPtr);
```
The two paths (host-upload for tests, device-ptr for real pipeline) can coexist behind a private variant or be refactored at that point.

---

## Verification Checklist

- [ ] `ofa-test.exe` prints `[PASS]` on RTX 5070 Ti
- [ ] `ofa-test-output.png` shows uniform red/green (expected +8/+4 vectors) in the central region
- [ ] `openxr-api-layer` project still builds clean (no new errors from NvOF include path)
- [ ] No NvOf headers leak into `pch.h`
