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

## Phase 3 — Decoupled Runtime Pacing Architecture (Asynchronous Presentation)

**Objective:** Replace failed dual-submit assumptions with a runtime-compliant decoupled model that owns headset-rate pacing on a dedicated layer thread.

### Tasks
1. Implement an **independent runtime submission thread** in the layer that continuously drives:
   - `xrWaitFrame -> xrBeginFrame -> xrEndFrame`
   - one runtime submit opportunity per runtime cadence tick.
2. Keep the **application thread unthrottled**:
   - do not artificially block app-facing `xrWaitFrame`
   - map app-produced frames to the next eligible predicted display time in layer-managed state.
3. Implement the **Holding Pen (ring buffer)** in `FrameBroker`:
   - on app `xrEndFrame`, capture frame payload + timing metadata into a bounded queue
   - return `XR_SUCCESS` to app without immediately forwarding submit to runtime thread path.
4. Implement runtime-thread **dynamic decision policy** at each submit deadline:
   - Path A (on-time): if a fresh app frame is queued, submit it unmodified
   - Path B (deadline miss): synthesize from buffered history and submit synthetic output.
5. Implement **fractional Δt motion scaling**:
   - compute scaling from source frame timestamps to runtime target `predictedDisplayTime`
   - feed scaled vectors/pose-warp inputs into synthesis path.
6. Add pacing instrumentation:
   - app `predictedDisplayTime`
   - app frame enqueue time
   - runtime thread `predictedDisplayTime`
   - synthesis completion time
   - actual runtime submit time.
7. Validate runtime acceptance/compositor stability under decoupled mode, including queue pressure, stale-frame behavior, and fallback correctness.

### Critical Notes
- Dual-submit from a single app frame is now considered a rejected approach for this project path.
- The runtime thread must remain OpenXR-order compliant and avoid illegal overlap/reentrancy across frame loops.
- Ring buffer policy must be explicit (drop-oldest vs drop-newest) and observable via tracing.

### Exit Criteria
- Dedicated runtime submission loop is stable at headset cadence.
- App thread remains unblocked and no longer directly dictates runtime submission cadence.
- On-time path and synthesis fallback path both produce stable output without runtime state/validation errors.
- Fractional Δt scaling is active and trace-verified under variable app framerate.

---

## Phase 4 — Depth + Pose Correctness Validation

**Objective:** Eliminate 3D instability artifacts in-headset.

### Tasks
1. Validate and lock depth policy:
   - reversed-Z handling
   - near/far linearization
   - depth range consistency.
2. Validate pose-delta direction/sign in pre-warp (no inversion).
3. If supported in path, include synthesized depth submission and observe runtime reprojection effects.

### Exit Criteria
- No major parallax inversion/background drift.
- Near geometry (cockpit/interior) stable under head motion.

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

1. Item 10 lifecycle + rewrite completion.
2. Full live GPU stage wiring.
3. Decoupled runtime pacing architecture (independent runtime thread + buffered/synth fallback).
4. Depth/pose correctness.
5. Packaging + in-game execution.

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
