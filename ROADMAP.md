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

## Phase 3 — Decoupled Runtime Pacing with Total Resource Ownership

**Objective:** Replace failed metadata-queue assumptions with a controlled architecture where the layer owns synchronization guarantees and cross-thread image lifetime.

### Phase 3A — Color-Only Decoupled Runtime Thread (No Synthesis Yet)

**Objective:** Prove stable decoupled pacing first, with color-only forwarding and no depth complexity.

#### Tasks
1. Add a **Vulkan API companion interception prerequisite**:
   - intercept `vkCreateDevice` to force/verify `timelineSemaphore = VK_TRUE`
   - intercept `vkCmdPipelineBarrier` to sanitize invalid depth aspect masks and illegal layout transitions before they poison layer-managed submission paths.
2. Add **capability gate + safe fallback** before starting decoupled mode:
   - if Vulkan interception is unavailable, device creation path is not controllable, or required extensions/features are missing, force `passthrough` mode
   - never start the decoupled runtime thread in an uncontrolled environment.
3. Redefine the Holding Pen as **layer-owned images**, not metadata pointers:
   - allocate private layer-owned Vulkan images for queued color frames
   - on app `xrEndFrame`, issue `vkCmdCopyImage` (or equivalent compute copy) from app image into layer-owned image
   - only copied layer-owned images are allowed to cross the app-thread/runtime-thread boundary.
4. Implement dedicated runtime thread pacing loop:
   - runtime thread owns `xrWaitFrame -> xrBeginFrame -> xrEndFrame`
   - app thread remains unthrottled and enqueues copied color frames + timing metadata.
5. Make Phase 3 explicitly **color-only**:
   - strip or ignore `XrCompositionLayerDepthInfoKHR` chains in decoupled mode during 3A
   - submit color-only projection layers until decoupled pacing is proven stable.
6. Add **graceful synchronization teardown**:
   - on `xrDestroySession`, stop runtime thread deterministically
   - wait for queue/device idle (`vkQueueWaitIdle` or `vkDeviceWaitIdle`) before freeing layer-owned images
   - prevent `vkFreeMemory while in use` during shutdown.
7. Add instrumentation and observability:
   - capability-gate decision reason
   - app enqueue timestamp + frame id
   - runtime predicted display time + submit timestamp
   - queue depth and drop policy actions (drop-oldest/drop-newest).

#### Exit Criteria
- Decoupled runtime thread runs stably at headset cadence in color-only mode.
- App thread stays unblocked and no runtime validation/state errors occur.
- No use-after-free or memory-in-use teardown failures on session shutdown.
- Capability-gate fallback reliably reverts to passthrough on unsupported paths.

### Phase 3B — Synthesis-on-Demand in Decoupled Mode (Still Color-Only)

**Objective:** Add gap-filling synthesis to the proven 3A pacing architecture without reintroducing depth complexity.

#### Tasks
1. Add runtime-thread deadline decision policy:
   - Path A: submit fresh queued real frame when available
   - Path B: synthesize from recent layer-owned color history when deadline would be missed.
2. Implement **fractional Δt scaling** for motion vectors and pose warp:
   - compute from buffered frame timestamps to runtime target `predictedDisplayTime`
   - trace both raw and scaled Δt values for validation.
3. Keep synthesis strictly color-only in 3B:
   - no depth chain submission/re-attachment in this phase.
4. Validate compositor behavior under mixed real/synthetic color-only submits.

#### Exit Criteria
- On-time and synthesis fallback paths are both stable in decoupled mode.
- Fractional Δt scaling is active, traceable, and visually consistent under variable app framerate.
- No regression in frame pacing or synchronization stability from 3A baseline.

---

## Phase 4 — Depth Ownership + Depth-Chain Reintegration

**Objective:** Add depth back only after Phase 3 color-only decoupled mode is production-stable.

### Tasks
1. Implement layer-owned depth resource path:
   - allocate private depth images compatible with runtime submission requirements
   - deep-copy app depth (`D32_SFLOAT` and supported formats) into layer-owned depth images aligned with color holding-pen frames.
2. Reinstate depth extension chains on layer-submitted frames:
   - attach valid `XrCompositionLayerDepthInfoKHR` chains to real and synthetic submissions
   - ensure chain lifetime ownership is fully inside layer-managed memory for call scope.
3. Validate depth convention and correctness in decoupled mode:
   - reversed-Z handling
   - near/far linearization
   - depth range consistency across real and synthetic paths.
4. Validate pose-delta direction/sign in pre-warp with depth-enabled synthesis.
5. Validate compositor/runtime behavior with depth-enabled decoupled submission.

### Exit Criteria
- Depth chains are stable and valid on decoupled real/synthetic submissions.
- No major parallax inversion/background drift.
- Near geometry (cockpit/interior) remains stable under head motion with depth enabled.

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

1. Phase 3A color-only decoupled runtime pacing with layer-owned images.
2. Phase 3B synthesis-on-demand in decoupled color-only mode.
3. Phase 4 depth ownership + depth-chain reintegration.
4. Packaging + in-game execution.

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
