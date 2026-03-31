# OpenXR Motion Smoothing Layer — Claude Context

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

**Not yet implemented (stubs/TODOs):**
- Vulkan compute pipeline in VulkanFrameProcessor
- CUDA/OFA integration (NVOf SDK)
- 6DoF pose capture and usage (pre-warp, LSR)
- Depth acquisition (XR_KHR_composition_layer_depth or alternative)
- Actual frame synthesis (warp, blend, hole fill)
- Frame injection back to compositor via modified `xrEndFrame`

---

## File Map

```
openxr-api-layer/
  layer.cpp               Main layer logic — OpenXR hooks live here
  layer.h                 Layer class definition
  pch.h / pch.cpp         Precompiled headers
  framework/
    dispatch.h/.cpp       OpenXR function dispatch (template framework)
    entry.cpp             DLL entry + loader negotiation (template framework)
    log.h/.cpp            TraceLogging provider + macros
    util.h                Formatting utilities (poses, FOV, vectors)
  utils/
    general.h/.cpp        General utilities — safe to use and extend
    inputs.h / input.cpp  Input handling utilities
    composition.cpp       Composition layer utilities
    d3d11.cpp             DEAD CODE — template leftover, do not touch
    d3d12.cpp             DEAD CODE — template leftover, do not touch
    graphics.h            DEAD CODE — template leftover, do not touch
external/
  OpenXR-SDK/             Khronos OpenXR SDK (submodule)
  OpenXR-SDK-Source/      Khronos OpenXR SDK source (submodule)
  OpenXR-MixedReality/    Microsoft OpenXR MixedReality (submodule)
scripts/
  Install-Layer.ps1       Registers the layer in the Windows registry
  Uninstall-Layer.ps1     Unregisters the layer
  Tracing.wprp            WPR trace capture profile
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
```powershell
# Init git submodules (OpenXR-SDK, OpenXR-SDK-Source, OpenXR-MixedReality)
git submodule update --init --recursive
```

## Build System
- Solution file: SMOOTHING-OPENXR-LAYER.sln (project root)
- MSBuild must be invoked via powershell.exe due to Git Bash path handling
- NuGet packages: fmt, OpenXR.Headers, OpenXR.Loader, WIL (restore automatically)
- Build targets: Debug/Release × Win32/x64
- Primary output: bin\x64\Debug\ or bin\x64\Release\

## Build Commands
# Debug build
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Debug /p:Platform=x64 /m"

# Release build
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Release /p:Platform=x64 /m"

# Clean
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /t:Clean /p:Configuration=Debug /p:Platform=x64"

## Known Warnings
- LNK4099: PDB 'fmt.pdb' not found — benign, fmtd.lib ships without debug symbols. Ignore this.

## Build Rules
- Always build after making changes — never commit code that does not compile
- Fix all errors before committing, warnings are acceptable
- Run debug build for development, release build before merging to main

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
