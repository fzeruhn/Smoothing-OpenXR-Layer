# Development Roadmap (In-Game Validation Track)

This roadmap replaces the old item-by-item sequence with a phased execution plan focused on getting a **buildable, testable API-layer DLL** for Star Citizen validation as fast and safely as possible.

Critical architecture constraints are baked in:
- This project builds an **OpenXR API layer DLL**, **not** `openxr_loader.dll`.
- OpenXR frame pacing is strict (`xrWaitFrame -> xrBeginFrame -> xrEndFrame`), so dual-submit/FG pacing must be treated as a dedicated architecture phase, not a quick extension of single-submit rewrite.

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

## Phase 3 — Timing Semantics Validation (Decoupled Presentation & Asynchronous Submission.)

**Objective:** Validate runtime-compliant pacing semantics before enabling any 45->90 behavior.

### Tasks
1. Stabilize **single-submit rewritten path** first.
2. Add instrumentation:
   - app `predictedDisplayTime`
   - synthesis completion timestamp
   - actual submission timestamp
3. Prototype dual-submit behavior behind runtime flag **only after single-submit stability**.
4. Implement a buffering strategy in FrameBroker to hold N past frames. Calculate fractional motion vector scaling based on Δt between the buffered frames and the target predictedDisplayTime.
4. Validate runtime acceptance/compositor stability/frame pacing under that mode.

### Critical Note
Naive “two submits from one app frame” is likely non-compliant for many runtimes. Treat dual-submit as experimental and heavily gated until proven runtime-safe.

### Exit Criteria
- Single-submit path production-stable.
- Dual-submit behavior characterized and gated by runtime compatibility.

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
   - optional `dual-submit` (gated).
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
3. Timing semantics validation (single-submit first, dual-submit optional/gated).
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

