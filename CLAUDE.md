# OpenXR Motion Smoothing Layer — Claude Context

## Session Start Checklist

At the start of every session, run `git submodule status`. If any line begins with `-`, the submodules are not initialized — run `git submodule update --init --recursive` before doing anything else. This is a known issue with new worktrees and will cause the pre-build code generation step to fail if skipped.

## Project Identity

This is a custom OpenXR API layer implementing high-performance motion smoothing for VR. It sits between the OpenXR runtime and the application, intercepting `xrEndFrame` to synthesize intermediate frames before they reach the compositor. The goal is to meaningfully surpass ASW/SteamVR motion smoothing by fully exploiting Blackwell OFA, depth-guided warping, and foveated processing.

**Hardware targets:**
- HMD: Pimax Dream Air (4K per eye, 90Hz)
- GPU: NVIDIA RTX 5070 Ti (Blackwell architecture)

**Primary validation target:** Star Citizen on Vulkan

**This project is Vulkan-only.** There are no plans for D3D11/D3D12 support.

See `ARCHITECTURE.md` for full system design and `ROADMAP.md` for build sequence.

---

## Toolchain

| Tool | Version |
|---|---|
| Visual Studio | 2026 |
| CUDA Toolkit | 13.2 |
| NVOf SDK | 5.0.7 |
| Windows SDK | 10.0 |
| Target OS | Windows 11 |

Dependencies managed via NuGet (OpenXR headers/loader, fmt, WIL). CUDA and NVOf SDK are external installs, not NuGet packages.

---

## Current Implementation Status

**Implemented (layer.cpp):**
- `xrCreateSession` — detects Vulkan API, captures VkInstance, VkPhysicalDevice, VkDevice, queue family index
- `xrCreateSwapchain` — tracks color and depth swapchains separately; initializes `VulkanFrameProcessor` when color swapchain is created
- `xrEnumerateSwapchainImages` — maps VkImage handles per swapchain
- `xrAcquireSwapchainImage` — tracks active image index per swapchain
- `xrEndFrame` — captures current color/depth VkImages, dispatches async GPU processing stub, maintains frame history (`m_prevColor`, `m_prevDepth`)
- `VulkanFrameProcessor` — Vulkan command pool/buffer setup; compute pipeline is stubbed with TODOs

**Implemented (vulkan_cuda_interop — Roadmap Item 1 complete):**
- `SharedImage` — RAII wrapper; allocates VkImage with external memory flags, exports Win32 handle, imports into CUDA as `CUarray` mappable via surface objects
- `SharedSemaphore` — RAII wrapper; creates VkSemaphore with export flags, imports into CUDA as `CUexternalSemaphore`; exposes `signal(stream)` / `wait(stream)`
- Validated end-to-end by `interop-test`: CUDA kernel writes pattern to shared image, Vulkan reads back and verifies every pixel → **[PASS]**

**Implemented (ofa_pipeline — Roadmap Item 3 complete):**
- `OFAPipeline` — RAII wrapper around NvOF 5.0.7 C API; loads `nvofapi64.dll` dynamically, creates OFA instance, allocates GPU buffers, runs optical flow estimation
- `loadFrame(slot, hostGray8)` — uploads 8-bit grayscale frame to OFA input buffer (slot 0 = inputFrame/current, slot 1 = referenceFrame/previous)
- `execute()` — runs `nvOFExecute` and copies output motion vectors to host staging buffer; caller must `cuCtxSynchronize()` before reading `outputData()`
- Validated end-to-end by `ofa-test`: 256×256 random-noise frames with deterministic +8px/+4px shift; 2500/2500 central vectors within ±2 S10.5 units → **[PASS]**
- **NvOF FORWARD convention:** output vectors give displacement from `inputFrame` → `referenceFrame` (NOT reference → input). Expected flow for a +8,+4px shift with current=inputFrame is (-256, -128) in S10.5.

**Implemented (frame_synthesizer — Roadmap Item 7 complete):**
- `FrameSynthesizer` — RAII wrapper around bidirectional frame synthesis (atomic scatter + bilinear gather)
- Two-pass CUDA pipeline: depth-sorted atomic scatter (closest depth wins) → bilinear gather + 50/50 blend
- Validated end-to-end by `synthesis-test`: 256×256 checkerboard +16/+8 px translation, 100% central pixels within ±2/channel → **[PASS]**
- **Known integration point — Reversed-Z depth:** Most modern engines (including likely Star Citizen) use Reversed-Z depth for precision (1.0 = near, 0.0 = far). The scatter kernel assumes standard depth (0.0 = near). If depth acquisition yields inverted visuals (background occludes foreground), invert depth in `kernel_scatter`: `depth = 1.0f - depth` before packing. This is a **TODO for live integration testing**.

**Not yet implemented (stubs/TODOs):**
- Vulkan compute pipeline in VulkanFrameProcessor
- 6DoF pose capture and usage (pre-warp, LSR) — Roadmap Item 2
- Depth acquisition (XR_KHR_composition_layer_depth or alternative) — Roadmap Item 5
- Pre-OFA pose pre-warp (homography) — Roadmap Item 4
- Frame injection back to compositor via modified `xrEndFrame` — Roadmap Item 10
- Hole filling (edge-directed interpolation) — Roadmap Item 9

**OFA Pipeline: Deferred Optimizations (Future Items 4+)**

The current `OFAPipeline` is validated as a standalone, synchronous component. For VR integration, the following architectural improvements are planned but deferred to Items 4+ (pre-warp, frame synthesis):

- **Async Memory Copies:** Item 4 will switch from `cuMemcpy2D` (blocking) to `cuMemcpy2DAsync` with CUDA streams to keep GPU work overlapped with CPU work during the 11ms frame budget.
- **Variable Input Pitch:** Item 4 will add a `hostPitch` parameter to `loadFrame()` to handle Vulkan swapchain texture padding/stride, instead of assuming tightly-packed host data.
- **Zero-Copy GPU Pipeline:** Items 6-7 will bypass `loadFrame()` entirely and feed Vulkan textures directly into OFA via CUDA/Vulkan interop (`cudaGraphicsResource`), keeping frames GPU-resident throughout the pipeline.

---

## File Map

```
openxr-api-layer/
  layer.cpp                    Main layer logic — OpenXR hooks live here
  layer.h                      Layer class definition
  vulkan_cuda_interop.h/.cpp   SharedImage + SharedSemaphore RAII wrappers (Roadmap Item 1)
  frame_synthesizer.h/.cu      FrameSynthesizer RAII + CUDA kernels (Roadmap Item 7)
  pch.h / pch.cpp              Precompiled headers
  framework/
    dispatch.h/.cpp            OpenXR function dispatch (template framework)
    entry.cpp                  DLL entry + loader negotiation (template framework)
    log.h/.cpp                 TraceLogging provider + macros
    util.h                     Formatting utilities (poses, FOV, vectors)
  utils/
    general.h/.cpp             General utilities — safe to use and extend
    inputs.h / input.cpp       Input handling utilities
    composition.cpp            Composition layer utilities
    d3d11.cpp                  DEAD CODE — template leftover, do not touch
    d3d12.cpp                  DEAD CODE — template leftover, do not touch
    graphics.h                 DEAD CODE — template leftover, do not touch
interop-test/
  main.cpp                     Headless Vulkan + CUDA round-trip test harness
  fill_pattern.cu              CUDA kernel + C-callable launcher
  interop-test.vcxproj         Standalone console app; links cuda.lib + cudart.lib
ofa-test/
  main.cpp                     OFA end-to-end test: random noise frames, shift validation, PNG output
  stb_image_write.h            Header-only PNG writer
  ofa-test.vcxproj             Standalone console app; uses CUDA 13.2 build customizations
synthesis-test/
  main.cpp                     Bidirectional synthesis end-to-end test: checkerboard +16/+8 shift, validation
  synthesis-test.vcxproj       Standalone console app; uses CUDA 13.2 build customizations
external/
  OpenXR-SDK/                  Khronos OpenXR SDK (submodule)
  OpenXR-SDK-Source/           Khronos OpenXR SDK source (submodule)
  OpenXR-MixedReality/         Microsoft OpenXR MixedReality (submodule)
  PVR/                         Pimax SDK, to be used for better eye tracking data
scripts/
  Install-Layer.ps1            Registers the layer in the Windows registry
  Uninstall-Layer.ps1          Unregisters the layer
  Tracing.wprp                 WPR trace capture profile
```

**New systems** (OFA, warpers, hole fill, frame pacing, etc.) should go in **new dedicated files**, not added to `layer.cpp`. `layer.cpp` should remain focused on OpenXR hook implementations that delegate to subsystem classes.

---

## Code Conventions

**Language:** C++17/C++20 preferred. Use structured bindings, `std::optional`, `if constexpr`, `[[nodiscard]]`, etc. where appropriate.

**Hot path discipline:** `xrEndFrame` is on the ~11ms frame budget (90Hz). On this path:
- No heap allocations — use pre-allocated buffers
- No blocking locks — prefer async dispatch
- No synchronous CPU waits — queue work and return

**Error handling:** All Vulkan, CUDA, and OpenXR return codes must be checked. Use `CHECK_VKCMD`, `CHECK_XRCMD` macros where they exist. Do not silently discard error results.

**CUDA/Vulkan interop:** Use `VK_KHR_external_memory` + `VkSemaphore` for Vulkan↔CUDA image sharing. This pattern is used throughout the OFA pipeline.

**D3D code:** `utils/d3d11.cpp`, `utils/d3d12.cpp`, `utils/graphics.h` are **dead code** from the original template. Do not extend, reference, or maintain them.

---

## Build & Dev Setup

### Prerequisites
1. **Visual Studio 2026** with C++ desktop workload + NuGet package manager
2. **Python 3** in PATH (required for OpenXR header generation during build)
3. **CUDA 13.2 Toolkit** — install from NVIDIA (not via NuGet)
4. **NVOf SDK 5.0.7** — install from NVIDIA Optical Flow SDK

### First-time setup

## Build System
- Solution file: SMOOTHING-OPENXR-LAYER.sln (project root)
- MSBuild must be invoked via powershell.exe due to Git Bash path handling
- NuGet packages: fmt, OpenXR.Headers, OpenXR.Loader, WIL (restore automatically)
- Build targets: Debug/Release × Win32/x64
- Primary output: bin\x64\Debug\ or bin\x64\Release\

## Build Commands

# Debug build (always restore packages first)
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m"

# Release build
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Release /p:Platform=x64 /m"

# Clean
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /t:Clean /p:Configuration=Debug /p:Platform=x64"

## Build Rules
- Always build after making changes — never commit code that does not compile
- Fix all errors before committing, warnings are acceptable
- Run debug build for development, release build before merging to main

## Running the standalone tests

Both test exes live in `bin\x64\Debug\` (or `Release\`) and are run directly from the solution root:

```
bin\x64\Debug\interop-test.exe   # Vulkan/CUDA interop round-trip
bin\x64\Debug\ofa-test.exe       # OFA optical flow end-to-end
```

`ofa-test.exe` writes `ofa-test-output.png` to the current directory (color-coded flow field, red=X, green=Y, ±16px range).

### cudart64_13.dll
Both test exes require `cudart64_13.dll` at runtime. The DLL is **not** in the default PATH — copy it manually after the first build:
```
copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\cudart64_13.dll" bin\x64\Debug\
copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\cudart64_13.dll" bin\x64\Release\
```
Do **not** use `cudart_static.lib` in projects built with `/MDd` — the CRT mismatch (`/MT` vs `/MDd`) causes a pre-`main()` crash with no output.

### nvofapi64.dll
`ofa-test.exe` loads `nvofapi64.dll` at runtime via `LoadLibraryA`. This DLL ships with the NVIDIA driver and lives in `C:\Windows\System32\` — no manual copying needed.

### Install & Test
```powershell
# Register the layer with the OpenXR loader (run as admin or from elevated PS)
.\scripts\Install-Layer.ps1

# Uninstall
.\scripts\Uninstall-Layer.ps1
```

The layer descriptor JSONs are at `openxr-api-layer/openxr-api-layer.json` (x64) and `openxr-api-layer/openxr-api-layer-32.json` (x86).

### Tracing
Use `scripts/Tracing.wprp` with Windows Performance Recorder (WPR) to capture ETW traces. The layer emits TraceLogging events via `framework/log.h` macros.

---

## See Also

- **`ARCHITECTURE.md`** — Full design of all pipeline systems: OFA, warping, hole fill, frame pacing, foveation, LSR, frame submission
- **`ROADMAP.md`** — Ordered build sequence with dependency rationale
