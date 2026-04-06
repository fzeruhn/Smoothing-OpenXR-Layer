# OpenXR Motion Smoothing Layer — Claude Context

## Session Start Checklist

At the start of every session, run `git submodule status`. If any line begins with `-`, the submodules are not initialized — run `git submodule update --init --recursive` before doing anything else. This is a known issue with new worktrees and will cause the pre-build code generation step to fail if skipped.

---

## Project Identity

This is a custom OpenXR API layer implementing high-performance motion smoothing for VR. It sits between the OpenXR runtime and the application, intercepting `xrEndFrame` to synthesize intermediate frames before they reach the compositor. The goal is to meaningfully surpass ASW/SteamVR motion smoothing by fully exploiting Blackwell OFA, depth-guided warping, and foveated processing.

**Hardware:** Pimax Dream Air (4K/eye, 90Hz) · NVIDIA RTX 5070 Ti (Blackwell)

**Primary validation target:** Star Citizen on Vulkan

**This project is Vulkan-only.** No D3D11/D3D12. See `ARCHITECTURE.md` for system design and `ROADMAP.md` for build sequence.

---

## Toolchain

| Tool | Version |
|---|---|
| Visual Studio | 2026 |
| CUDA Toolkit | 13.2 |
| NVOf SDK | 5.0.7 |
| Windows SDK | 10.0 |
| Target OS | Windows 11 |

NuGet dependencies: fmt, OpenXR.Headers, OpenXR.Loader, WIL (restore automatically). CUDA and NVOf SDK are external installs.

---

## Current Implementation Status

**Implemented:**
- **layer.cpp** — xrCreateSession, xrCreateSwapchain, xrEnumerateSwapchainImages, xrAcquireSwapchainImage, xrWaitFrame, xrEndFrame; delegates to modular frame/pose/depth helpers. Runtime behavior still pass-through at final submission.
- **Item 1 — vulkan_cuda_interop** — `SharedImage` (VkImage↔CUarray) + `SharedSemaphore` (VkSemaphore↔CUexternalSemaphore). Validated by `interop-test` → **[PASS]**
- **Item 3 — ofa_pipeline** — `OFAPipeline`: NvOF 5.0.7 dynamic load, loadFrame/execute. **NvOF FORWARD convention:** vectors give displacement inputFrame→referenceFrame (NOT reference→input). Validated by `ofa-test` → **[PASS]**
- **Item 2 / 1.5 foundation** — `FrameContext` + `PoseProvider`: caches predicted display time from `xrWaitFrame`, populates predicted views via `xrLocateViews`, records render-view FOV/pose from projection layers.
- **Item 4 (infra) — pose_warp** — `PoseWarper` (backward CUDA warp) + `pose_warp_math` (homography from quaternion+FOV). Integration now has pose plumbing, but pre-warp is still not wired into live OFA path. Validated by `pose-warp-test` → **[PASS]**
- **Item 5 foundation** — `DepthProvider`: parses `XR_KHR_composition_layer_depth` from projection view chains, resolves Vulkan depth image handles, surfaces metadata (`minDepth/maxDepth/nearZ/farZ/reversedZ`) with fallback to tracked depth swapchain.
- **Item 7 — frame_synthesizer** — `FrameSynthesizer`: depth-sorted scatter + bilinear gather + 50/50 blend. **Reversed-Z note:** Star Citizen likely uses 1=near/0=far — invert depth in `kernel_scatter` if background occludes foreground. Validated by `synthesis-test` → **[PASS]**
- **Item 8 — stereo_vector_adapter** — `StereoVectorAdapter`: derives right-eye vectors from left-eye OFA via binocular disparity + depth. Validated by `stereo-adapter-test` → **[PASS]**
- **Item 9 — hole_filler** — `HoleFiller`: push-pull mipmap fill in-place; stable `fill(frame, holeMap)` interface (AI inpainting slot). Validated by `hole-fill-test` → **[PASS]**
- **Frame transport extraction** — `FrameBroker`: centralized swapchain registration, image mapping, acquired-index tracking, and current color/depth retrieval.
- **Injection scaffolding** — `FrameInjection`: creates dedicated synthetic color swapchain and exposes readiness; recursion guard added for internal swapchain creation.
- **Vulkan submission safety hardening** — `VulkanFrameProcessor` migrated from single command buffer to command-buffer ring + per-slot fence gating to avoid CPU/GPU re-record races.

**Pending (stubs/TODOs):**
- **EAC-Safe Sync Swap (Phase 3, Priority 1):** Rip out all `timelineSemaphore` dependencies. Implement thread-crossing sync using binary `VkSemaphore` + `VkFence` exclusively.
- **True Holding Pen / Layer-Owned Color Images (Phase 3, Priority 2):** Allocate private Vulkan color image ring buffer. Wire `vkCmdCopyImage` deep-copy in `xrEndFrame` so app thread returns `XR_SUCCESS` immediately; only layer-owned copies cross the thread boundary.
- **Decoupled Runtime Thread (Phase 3, Priority 3):** Spin up independent thread owning `xrWaitFrame -> xrBeginFrame -> xrEndFrame`. Submit layer-owned color-only images; strip all `XrCompositionLayerDepthInfoKHR` chains during stabilization.
- **Synthesis Fallback (Phase 3, Priority 4):** Wire motion smoothing (OFA → stereo adaptation → pre-warp → synthesis → hole fill) as the deadline-miss path in the runtime thread; use fractional Δt scaling against `predictedDisplayTime`.
- **Depth Ownership (Phase 4, Priority 5):** Allocate layer-owned `D32_SFLOAT` depth images, deep-copy alongside color, reconstruct valid `XrCompositionLayerDepthInfoKHR` chains pointing to layer-owned depth memory.
- **Item 2 integration remainder:** compute and use `pose_delta = display_pose * inverse(render_pose)` in live pre-warp path.
- **Item 5 integration remainder:** apply depth convention policy in synthesis path (reversed-Z + near/far handling validation in-game).
- **In-game validation pass:** verify Star Citizen runtime traces (`r_sterodepthcomposition=1`) show stable depth detection, synthesis readiness, active rewrite, and no fence starvation.

**Architecture corrections (must follow):**
- This project outputs an **OpenXR API layer DLL** referenced by manifest `library_path`; it must **not** be named/replaced as `openxr_loader.dll`.
- Treat OpenXR pacing semantics as strict (`xrWaitFrame -> xrBeginFrame -> xrEndFrame`). The decoupled runtime thread owns this loop; the app thread must never block on compositor pacing.
- **No Vulkan API interception.** Do not intercept `vkCreateDevice`, `vkCmdPipelineBarrier`, or any other Vulkan entry point. The layer is purely an OpenXR API layer — Vulkan companion layers are dropped for EAC safety.
- **No timeline semaphores.** All thread-crossing GPU synchronization uses binary `VkSemaphore` + `VkFence` only.
- **True Holding Pen:** the app's Vulkan memory is never accessed by the runtime thread. All cross-thread image traffic passes through deep-copied layer-owned images.

**OFA deferred optimizations (for live integration):** switch to `cuMemcpy2DAsync` (async copies); add `hostPitch` to `loadFrame()` (Vulkan stride); bypass `loadFrame()` entirely via CUDA/Vulkan interop to keep frames GPU-resident.

---

## File Map

```
openxr-api-layer/
  layer.cpp / layer.h              Main OpenXR hooks
  frame_context.h                  Per-frame pose/depth/render context
  pose_provider.h/.cpp             Predicted view/pose capture (xrWaitFrame + xrLocateViews)
  depth_provider.h/.cpp            XR_KHR_composition_layer_depth parsing + depth metadata
  frame_broker.h/.cpp              Swapchain/image/index tracking for frame transport
  frame_injection.h/.cpp           Synthetic swapchain allocation + injection readiness state
  vulkan_cuda_interop.h/.cpp       SharedImage + SharedSemaphore (Item 1)
  ofa_pipeline.h/.cpp              OFAPipeline + NvOF 5.0.7 (Item 3)
  pose_warp_math.h/.cpp            Homography from quaternion+FOV (Item 4)
  pose_warp.h/.cu                  PoseWarper CUDA kernel (Item 4)
  frame_synthesizer.h/.cu          FrameSynthesizer CUDA kernels (Item 7)
  stereo_vector_adapter.h/.cu      StereoVectorAdapter (Item 8)
  hole_filler.h/.cu                HoleFiller push-pull (Item 9)
  pch.h / pch.cpp                  Precompiled headers
  framework/                       dispatch, entry, log, util (OpenXR framework)
  utils/                           general, inputs, composition (safe to extend)
    d3d11.cpp / d3d12.cpp / graphics.h   DEAD CODE — template leftover, do not touch
interop-test/        Vulkan/CUDA round-trip test
ofa-test/            OFA end-to-end + PNG output
synthesis-test/      Bidirectional synthesis test
stereo-adapter-test/ Stereo vector adaptation test
hole-fill-test/      Push-pull hole filling test
pose-warp-test/      Pose warp checkerboard test
external/            OpenXR-SDK, OpenXR-SDK-Source, OpenXR-MixedReality (submodules); PVR/ (Pimax SDK)
scripts/             Install-Layer.ps1, Uninstall-Layer.ps1, Tracing.wprp
```

New systems go in **new dedicated files**, not `layer.cpp`. `layer.cpp` stays focused on OpenXR hook implementations that delegate to subsystem classes.

---

## Code Conventions

**Language:** C++17/C++20. Use structured bindings, `std::optional`, `if constexpr`, `[[nodiscard]]`, etc.

**Hot path (`xrEndFrame`, ~11ms budget):**
- No heap allocations — use pre-allocated buffers
- No blocking locks — prefer async dispatch
- No synchronous CPU waits — queue work and return

**Error handling:** Check all Vulkan/CUDA/OpenXR return codes. Use `CHECK_VKCMD`, `CHECK_XRCMD` macros. Never silently discard errors.

**CUDA/Vulkan interop:** Use `VK_KHR_external_memory` + `VkSemaphore` for image sharing.

**D3D code:** `d3d11.cpp`, `d3d12.cpp`, `graphics.h` — dead code from original template. Do not extend, reference, or maintain them.

---

## Build & Dev Setup

**Prerequisites:**
1. Visual Studio 2026 (C++ desktop workload + NuGet)
2. Python 3 in PATH (OpenXR header generation during build)
3. CUDA 13.2 Toolkit (external install)
4. NVOf SDK 5.0.7 (external install)

First-time: run `git submodule update --init --recursive` before building.

**Build commands (invoke via powershell.exe — Git Bash path handling):**
```powershell
# Debug (restore NuGet first)
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m"

# Release
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Release /p:Platform=x64 /m"

# Clean
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /t:Clean /p:Configuration=Debug /p:Platform=x64"
```

Output: `bin\x64\Debug\` or `bin\x64\Release\`. Always build after changes; fix all errors before committing.

**Runtime DLLs — copy after first build:**
```
copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\cudart64_13.dll" bin\x64\Debug\
copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\cudart64_13.dll" bin\x64\Release\
```
Do **not** use `cudart_static.lib` with `/MDd` — CRT mismatch causes a pre-`main()` crash. `nvofapi64.dll` ships with the NVIDIA driver (`C:\Windows\System32\`), no copy needed.

**Standalone tests** (run from solution root):
```
bin\x64\Debug\interop-test.exe        # Vulkan/CUDA interop round-trip
bin\x64\Debug\ofa-test.exe            # OFA optical flow (writes ofa-test-output.png)
bin\x64\Debug\synthesis-test.exe      # Bidirectional frame synthesis
bin\x64\Debug\hole-fill-test.exe      # Push-pull hole filling
bin\x64\Debug\pose-warp-test.exe      # Pose-based homography warp
bin\x64\Debug\stereo-adapter-test.exe # Stereo vector adaptation
```

**Install / Uninstall** (run elevated):
```powershell
.\scripts\Install-Layer.ps1    # Register with OpenXR loader
.\scripts\Uninstall-Layer.ps1  # Unregister
```
Layer descriptor JSONs: `openxr-api-layer/openxr-api-layer.json` (x64) and `openxr-api-layer/openxr-api-layer-32.json` (x86).

**Tracing:** Use `scripts/Tracing.wprp` with WPR. The layer emits TraceLogging events via `framework/log.h` macros.
