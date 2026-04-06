# Development Roadmap (In-Game Validation Track)

This roadmap replaces the old item-by-item sequence with a phased execution plan focused on getting a **buildable, testable API-layer DLL** for Star Citizen validation as fast and safely as possible.

Critical architecture constraints are baked in:
- This project builds an **OpenXR API layer DLL**, **not** `openxr_loader.dll`.
- OpenXR frame pacing is strict (`xrWaitFrame -> xrBeginFrame -> xrEndFrame`), so asynchronous presentation must be implemented as a dedicated decoupled architecture phase, not as naive dual-submit from one app frame.

---

## Completed Foundation (Current State)

- **Interop validated**: Vulkan/CUDA memory + semaphore interop (`vulkan_cuda_interop.*`) with `interop-test` pass.
- **Core synthesis components validated headless**:
  - OFA pipeline (`ofa_pipeline.*`) — pass
  - Pose warp infra (`pose_warp.*`, `pose_warp_math.*`) — pass
  - Frame synthesizer (`frame_synthesizer.*`) — pass
  - Stereo adapter (`stereo_vector_adapter.*`) — pass
  - Hole filler (`hole_filler.*`) — pass
- **OpenXR integration scaffolding landed**:
  - `FrameContext`, `PoseProvider`, `DepthProvider`
  - `FrameBroker` (swapchain/image/index tracking)
  - `FrameInjection` (synthetic swapchain allocation readiness)
  - `xrWaitFrame` hook + predicted view capture path
- **Submission safety hardening**:
  - `VulkanFrameProcessor` command-buffer ring + fence gating (replaces unsafe single-buffer reuse)

What is still missing: end-to-end in-game submission rewrite with real synthesized image writes and full live GPU stage wiring.

---

## Phase 0 — Stabilize Build + Runtime Baseline | Complete ✅

**Objective:** Ensure reproducible “known-good” pass-through baseline before turning on rewrite/injection.

### Tasks
1. Confirm **Release** build of `openxr-api-layer` produces the API-layer DLL referenced by manifest `library_path` (do **not** target `openxr_loader.dll`).
2. Verify install/uninstall scripts still register the correct JSON and DLL output location.
3. Smoke test in runtime: layer loads, no rewrite active, no validation/state errors.

### Exit Criteria
- Clean Release build.
- Layer loads through OpenXR loader via manifest registration.
- Pass-through rendering remains unchanged.

---

## Phase 1 — Complete Item 10 Core Submission Path (Single-Submit First) | Complete ✅

**Objective:** Finish safe synthetic swapchain submission pipeline with one rewritten submit per app frame.

### Tasks
1. **Synthetic image lifecycle**
   - In `xrEndFrame` active path: Acquire -> Wait -> Write -> Release synthetic swapchain image.
   - Explicit fallback to original `frameEndInfo` on any lifecycle failure.
2. **Actual image write path**
   - Replace readiness/copy placeholder with real synthesized output write.
   - Maintain explicit Vulkan layout transitions before write and before compositor consumption.
3. **Deep-copy rewrite of frame structures**
   - Clone `XrFrameEndInfo` and projection layer/view structures into owned memory for call scope.
   - Rewrite projection `subImage.swapchain` (and corresponding index/rect handling) to synthetic swapchain.
   - Preserve non-projection layers unchanged.
4. **Rewrite guards**
   - Rewrite only when synthesis output for that frame is valid and synchronized.
   - Otherwise submit original untouched `frameEndInfo`.

### Exit Criteria
- Injection rewrite path active and stable.
- No invalid handle/state/runtime validation errors.
- No crashes from pointer lifetime/struct ownership issues.

---

## Phase 2 — GPU Pipeline Wiring Completion (`VulkanFrameProcessor`) | Complete ✅

**Objective:** Complete the live synthesis chain actually consumed by Phase 1 submission rewrite.

### Tasks
1. Integrate live stage order:
   - pre-warp -> OFA -> depth linearization policy -> stereo adaptation -> synthesis -> hole fill.
2. Enforce Vulkan/CUDA synchronization per stage with explicit ownership boundaries.
3. Keep ring/fence flow non-blocking (`xrEndFrame` hot path must not CPU-stall).
4. Emit per-stage validity flags and only expose fully-valid output to submission path.

### Exit Criteria
- Full synthesis executes on live frames.
- Stable frame-to-frame resource ownership and synchronization.
- No fence starvation in steady state traces.

---

## Phase 3 — Decoupled Runtime Pacing & Owned-Resource Architecture

**Objective:** Replace failed dual-submit and metadata assumptions with a fully isolated OpenXR layer that owns its own Vulkan memory and synchronizes via EAC-safe binary primitives.

### Tasks
1. **Implement EAC-Safe Synchronization:**
   - Rip out all `timelineSemaphore` dependencies.
   - Implement thread-crossing synchronization using strict **Binary Semaphores** and **Vulkan Fences**.
2. **Implement the "True Holding Pen" (Layer-Owned Images):**
   - Allocate a private ring buffer of layer-owned Vulkan color images.
   - On app `xrEndFrame`, execute a deep copy (`vkCmdCopyImage` or compute shader) from the app's swapchain into the layer's private image.
   - Return `XR_SUCCESS` to the app immediately, leaving the app thread unthrottled.
3. **Implement the Independent Runtime Submission Thread:**
   - Runtime thread owns the `xrWaitFrame -> xrBeginFrame -> xrEndFrame` loop to the compositor.
   - The thread only ever submits **layer-owned copied images**, completely isolating SteamVR from the application's memory state.
4. **Enforce Color-Only Decoupling (Phase 3 Stabilization):**
   - Strip all `XrCompositionLayerDepthInfoKHR` chains during this phase.
   - Accept external depth-barrier validation spam temporarily, knowing the compositor path is isolated and safe.
5. **Implement Runtime-Thread Decision Policy & Synthesis:**
   - Path A (on-time): Submit the freshest copied frame.
   - Path B (deadline miss): Synthesize from the layer-owned buffer using fractional Δt motion scaling.
6. **Graceful Teardown:**
   - On shutdown, ensure `vkDeviceWaitIdle` or `vkQueueWaitIdle` is called to drain binary semaphores before freeing the layer's private images.

### Exit Criteria
- Decoupled runtime loop is stable at headset cadence using binary sync primitives.
- App thread remains unblocked.
- SteamVR/Compositor receives valid, layer-owned color images without fatal crashes.

---

## Phase 4 — Depth Ownership + Depth-Chain Reintegration

**Objective:** Safely re-introduce depth data into the stable, decoupled Phase 3 pipeline.

### Tasks
1. **Implement Layer-Owned Depth Holding Pen:**
   - Allocate private `D32_SFLOAT` (or equivalent) Vulkan images.
   - Deep-copy app depth data into layer-owned images concurrently with color data.
2. **Sanitize & Re-attach Depth Chains:**
   - Reconstruct valid `XrCompositionLayerDepthInfoKHR` chains pointing to *layer-owned* depth memory.
   - Submit real and synthetic frames with the new depth payload.
3. Validate depth convention correctness:
   - reversed-Z handling, near/far linearization, and pose-delta direction.

### Exit Criteria
- No major parallax inversion/background drift.
- Validation spam related to depth barriers is resolved or safely ignored by the compositor.

---

## Phase 5 — In-Game Readiness Package

**Objective:** Produce reproducible build/install/test loop for rapid iteration in Star Citizen.

### Tasks
1. Release package: API layer DLL + manifests + install scripts.
2. Confirm one-command install/uninstall flow.
3. Add runtime toggles:
   - `passthrough`
   - `injection-copy`
   - `full-synthesis`
   - `decoupled-runtime-thread` (gated)
4. Lock test logging profile categories:
   - depth
   - fences/sync
   - submission rewrite
   - timing.

### Exit Criteria
- Reproducible install + launch workflow.
- Toggle-based fallback for rapid triage in headset.

---

## In-Game Test Plan (Star Citizen)

1. Launch offline / EAC-safe environment.
2. Enable `r_sterodepthcomposition=1`.
3. Test sequence:
   - pass-through baseline
   - injection-copy
   - full-synthesis.
4. Required signals:
   - depth chain present/stable
   - synthesis active
   - injection rewrite active
   - fence stalls near zero
   - no `xrEndFrame` CPU hitching.

### Pass Criteria
- Visually stable headset output.
- No runtime validation/state errors.
- Sustained session without degradation.

---

## Strict Priority Order
*(Updated to reflect the new sequence)*

1. **EAC-Safe Sync Swap:** Replace timeline semaphores with binary semaphores/fences.
2. **True Holding Pen (Color):** Allocate layer-owned images and wire up the deep copy on `xrEndFrame`.
3. **Decoupled Runtime Thread:** Spin up the pacing loop to submit the layer-owned images (Color-only).
4. **Synthesis Fallback:** Wire up the motion smoothing on missed deadlines.
5. **Depth Ownership:** Copy depth buffers and rebuild the extension chains (Phase 4).
6. **Packaging + In-game Execution.**

---

## Key Files for Remaining Work

- `openxr-api-layer/layer.cpp`
- `openxr-api-layer/frame_injection.h/.cpp`
- `openxr-api-layer/frame_broker.h/.cpp`
- `openxr-api-layer/frame_context.h`
- `openxr-api-layer/depth_provider.h/.cpp`
- `openxr-api-layer/pose_provider.h/.cpp`
- `openxr-api-layer/frame_synthesizer.h/.cu`
- `openxr-api-layer/ofa_pipeline.h/.cpp`
- `openxr-api-layer/stereo_vector_adapter.h/.cu`
- `openxr-api-layer/hole_filler.h/.cu`
