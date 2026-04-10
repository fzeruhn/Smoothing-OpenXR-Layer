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
- **xrDestroySession interception:** RuntimeThread joined deterministically before session teardown; prevents use-after-free on session objects.
- **xrWaitSwapchainImage interception:** added to dispatch override list; early-exits for layer-owned primary color swapchain.
- **Phase 3A Task 4 — Private layer-owned primary color images (Complete ✅):**
  - `AllocatePrivateColorImages`: N device-local `VkImage`s (COLOR_ATTACHMENT + TRANSFER_SRC)
  - `xrEnumerateSwapchainImages`: detects primary color swapchain, substitutes private image handles before returning to app
  - `xrAcquireSwapchainImage` / `xrWaitSwapchainImage` / `xrReleaseSwapchainImage`: intercept primary color path — SteamVR's real swapchain is never acquired or released from the app thread
  - `TeardownPhase3Resources`: frees private images after `vkQueueWaitIdle` guard
- **Root cause identified:** concurrent `vkQueueSubmit` from RuntimeThread and app thread on the same `VkQueue` — Vulkan external-sync violation. The private image work above fixes swapchain cycling (a real but secondary bug); the queue race is the crash root cause and requires queue isolation to fix.

What is still missing: queue isolation (confirmed crash root cause), decoupled runtime thread cross-thread wiring, and full live GPU synthesis pipeline.

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
1. ~~Add a Vulkan API companion interception prerequisite~~ — **REMOVED**: intercepting `vkCreateDevice` is not EAC-safe and is not required. Queue isolation is achieved entirely via OpenXR-level hooks. No raw Vulkan entry-point detours of any kind.
2. Add **hardware queue isolation (EAC-safe, OpenXR-level only)** — *next active task*:
   - **Confirmed crash root cause:** RT and app thread call `vkQueueSubmit` on the same `VkQueue` handle concurrently — Vulkan external-synchronization violation. CPU mutex (`g_queueMutex`) cannot fix this because it cannot wrap the app's own Vulkan submit calls.
   - **Step A — probe (zero cost, try first):** in `xrCreateSession`, call `vkGetDeviceQueue(device, queueFamilyIndex, queueIndex+1, &rtQueue)`; if non-null and ≠ app queue, assign to RT and skip Steps B/C. Works when the app requested multiple queues (common in AAA engines).
   - **Step B — `xrCreateVulkanDeviceKHR` interception:** if called by the app, inspect `VkDeviceCreateInfo`; if the target queue family requests only 1 queue, bump `queueCount` to 2. Then in `xrCreateSession` rewrite `XrGraphicsBindingVulkanKHR.queueIndex` from 0 to 1 for the RT. Only fires if app uses `XR_KHR_vulkan_enable2` — Star Citizen may not use this extension.
   - **Step C — fallback policy:** if probe returns null AND `xrCreateVulkanDeviceKHR` was never intercepted, proxy RT's `vkQueueSubmit` calls through a submit functor dispatched on the app thread (RT prepares command buffer, app thread executes the submit at `xrReleaseSwapchainImage`), OR force `passthrough` mode.
   - Do **not** detour `vkQueueSubmit`, `vkQueuePresentKHR`, or `vkCreateDevice` — EAC-risky and out-of-bounds.
3. Add **capability gate + safe fallback** before starting decoupled mode:
   - log queue isolation outcome (VkQueue handle, family, index) at session creation
   - if no confirmed isolated queue exists (probe failed, no `xrCreateVulkanDeviceKHR` intercept, proxy not implemented), force `passthrough` mode
   - never start the decoupled runtime thread without confirmed queue isolation.
4. ~~Redefine the Holding Pen as layer-owned images~~ — **Complete ✅**: private `VkImage` ring allocated in `xrEnumerateSwapchainImages`; acquire/wait/release intercepts in place; `TeardownPhase3Resources` with idle guard. See Completed Foundation above.
5. Implement dedicated runtime thread pacing loop:
   - runtime thread owns `xrWaitFrame -> xrBeginFrame -> xrEndFrame`
   - app thread remains unthrottled and enqueues copied color frames + timing metadata.
6. Make Phase 3 explicitly **color-only**:
   - strip or ignore `XrCompositionLayerDepthInfoKHR` chains in decoupled mode during 3A
   - submit color-only projection layers until decoupled pacing is proven stable.
7. Add **graceful synchronization teardown**:
   - on `xrDestroySession`, stop runtime thread deterministically
   - wait for queue/device idle (`vkQueueWaitIdle` or `vkDeviceWaitIdle`) before freeing layer-owned images
   - prevent `vkFreeMemory while in use` during shutdown.
8. Add instrumentation and observability:
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

*(Updated to reflect confirmed root cause and current implementation state)*

1. **Hardware queue isolation (Phase 3A Task 2):** probe → `xrCreateVulkanDeviceKHR` interception → fallback policy. Confirmed crash root cause — nothing else should move forward until this is resolved.
2. **EAC-safe sync swap:** replace any remaining timeline-semaphore dependencies with binary semaphores/fences.
3. **Decoupled runtime thread wiring (Phase 3A Tasks 5–8):** ring-buffer cross-thread delivery, pacing loop, drop policy, instrumentation, graceful teardown.
4. **Synthesis fallback (Phase 3B):** wire motion smoothing (OFA → stereo → pre-warp → synthesis → hole fill) as deadline-miss path; fractional Δt scaling against `predictedDisplayTime`.
5. **Depth ownership (Phase 4):** layer-owned depth images, deep copy alongside color, reconstruct `XrCompositionLayerDepthInfoKHR` chains.
6. **Packaging + in-game execution (Phase 5).**

*~~True holding pen / layer-owned color images~~ — Complete ✅ (was Priority 3).*

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
