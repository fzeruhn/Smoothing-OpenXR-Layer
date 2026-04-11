# OpenXR Motion Smoothing Layer — Codex Context

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
- **layer.cpp** — xrCreateSession, xrCreateSwapchain, xrEnumerateSwapchainImages, xrAcquireSwapchainImage, xrWaitSwapchainImage, xrWaitFrame, xrEndFrame, xrDestroySession, xrPollEvent; delegates to modular frame/pose/depth helpers.
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
- **xrDestroySession** — intercepts to join `RuntimeThread` deterministically before session teardown, preventing use-after-free on session objects.
- **xrWaitSwapchainImage** — added to dispatch intercept list; early-exits for layer-managed primary color swapchain to match the synthesized acquire/release intercepts.
- **Private layer-owned primary color images (Phase 3A Task 4 — Complete)** — `AllocatePrivateColorImages` allocates N device-local `VkImage`s (COLOR_ATTACHMENT + TRANSFER_SRC); `xrEnumerateSwapchainImages` substitutes private images for the primary color swapchain before returning handles to the app; `xrAcquireSwapchainImage`/`xrWaitSwapchainImage`/`xrReleaseSwapchainImage` intercept primary color swapchain operations so SteamVR's real swapchain is never acquired or released from the app thread; `TeardownPhase3Resources` frees private images after `vkQueueWaitIdle` guard.

**Pending (stubs/TODOs):**
- **Item 2 integration remainder:** compute and use `pose_delta = display_pose * inverse(render_pose)` in live pre-warp path.
- **Item 5 integration remainder:** apply depth convention policy in synthesis path (reversed-Z + near/far handling validation in-game).
- **Item 10 remainder (major blocker for in-game FG):** write synthesized output into injection swapchain images and rewrite downstream projection-layer subimages in `xrEndFrame`.
- **VulkanFrameProcessor functional pipeline:** current dispatch path is infrastructure only; OFA → stereo adaptation → pre-warp → synthesis → hole-fill still needs live wiring.
- **EAC-safe sync migration (priority):** replace timeline-sem semaphore assumptions with binary semaphore/fence flow compatible with runtime + app ownership boundaries.
- **Hardware queue isolation (next priority — confirmed crash root cause):** two threads call `vkQueueSubmit` on the same `VkQueue` handle concurrently (RT blit + app renders) — Vulkan external-sync violation; `g_queueMutex` cannot fix this because it cannot wrap the app's own submits. Three-step fix: (A) probe `vkGetDeviceQueue(device, family, index+1)` in `xrCreateSession` — zero cost, works if app requested multiple queues; (B) intercept `xrCreateVulkanDeviceKHR` to bump `queueCount` to 2 when only 1 is requested, then rewrite `queueIndex` to 1 in `xrCreateSession`; (C) if both fail (app used raw `vkCreateDevice` with 1 queue — possible for Star Citizen), proxy RT queue submissions through the app thread or force passthrough.
- **Decoupled runtime thread (color-first):** layer-owned color images complete (Phase 3A Task 4 done). Remaining: queue isolation (prerequisite) + ring-buffer cross-thread delivery wiring. Synthesis fallback follows only after queue-isolated color-only path is stable.
- **In-game validation pass:** verify Star Citizen runtime traces (`r_sterodepthcomposition=1`) show stable depth detection, synthesis readiness, active rewrite, and no fence starvation.
- **Depth reintegration (post-Phase-3):** depth copy/ownership and `XrCompositionLayerDepthInfoKHR` re-attachment are Phase 4 work, not part of initial decoupled-thread bring-up.
- **Graceful teardown:** on `xrDestroySession`, deterministically stop decoupled thread and idle queue/device before freeing layer-owned images to avoid `vkFreeMemory while in use`.

**Architecture corrections (must follow):**
- This project outputs an **OpenXR API layer DLL** referenced by manifest `library_path`; it must **not** be named/replaced as `openxr_loader.dll`.
- Treat OpenXR pacing semantics as strict (`xrWaitFrame -> xrBeginFrame -> xrEndFrame`) on the runtime thread; do not use naive dual-submit from a single app heartbeat.
- Do **not** detour raw Vulkan submit/present entry points (`vkQueueSubmit`, `vkQueuePresentKHR`, `vkCreateDevice`) for any purpose — EAC-risky and out-of-bounds. Queue isolation must be achieved via OpenXR-level hooks (`xrCreateVulkanDeviceKHR`, `xrCreateSession`) only.
- Decoupled runtime submission is only valid when queue isolation and capability gates succeed; otherwise force safe passthrough.
- **Confirmed VK_ERROR_DEVICE_LOST root cause:** concurrent `vkQueueSubmit` from RuntimeThread and app thread on the same `VkQueue` handle — Vulkan external-synchronization violation. A CPU mutex cannot fix this because it cannot wrap the app's own Vulkan submit calls. Fix requires the RT to own a separate `VkQueue` handle.
- **No timeline semaphores.** All thread-crossing GPU synchronization uses binary `VkSemaphore` + `VkFence` only.

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
scripts/             Install-Layer.ps1, Uninstall-Layer.ps1, Reinstall-Layer.ps1, Tracing.wprp
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
.\scripts\Reinstall-Layer.ps1  # Uninstall+Install in one elevated pass
```
Layer descriptor JSONs: `openxr-api-layer/openxr-api-layer.json` (x64) and `openxr-api-layer/openxr-api-layer-32.json` (x86).

**hello_xr validation workflow (preferred for current Phase 3 work):**
```powershell
# hello_xr executable + required renderer arg
& "C:\Project\OpenXR-SDK-Source\build\src\tests\hello_xr\Debug\hello_xr.exe" -g vulkan
```
- Baseline run is **optional** (use when comparing regressions or validating install-state assumptions), not required for every iteration.
- Layer run: install layer, then execute the same command and compare behavior.
- Timeout rule: do not let `hello_xr` run indefinitely in automation. If it does not exit within a reasonable window (eg. 60-90s), terminate it programmatically and treat as `TIMEOUT/HANG` for that test pass.
- Layer log tail command:
```powershell
Get-Content "$env:LOCALAPPDATA\SMOOTHING-OPENXR-LAYER\SMOOTHING-OPENXR-LAYER.log" -Tail 40
```
- Treat `hello_xr` console output as low-signal; use the layer log + trace events as primary diagnostics.
- Install script caveat: `Install-Layer.ps1` prefers `openxr-api-layer.json` next to the script, then `..\bin\x64\Release\openxr-api-layer.json`. Running `scripts\Reinstall-Layer.ps1` from repo root checks Debug first, then Release.

**Tracing:** Use `scripts/Tracing.wprp` with WPR. The layer emits TraceLogging events via `framework/log.h` macros.
