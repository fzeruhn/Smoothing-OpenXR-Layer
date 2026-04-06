# Architecture: OpenXR Motion Smoothing Layer

## Pipeline Overview

```
App (Star Citizen / Vulkan)
        │
        ▼  xrEndFrame(frameEndInfo)
┌─────────────────────────────────────────────┐
│           OpenXR API Layer (this project)   │
│                                             │
│  ┌─ App Thread ────────────────────────┐    │
│  │  Frame Capture:                     │    │
│  │  color[L,R], depth[L,R],            │    │
│  │  pose_render, pose_display, timing  │    │
│  │           │                         │    │
│  │           ▼                         │    │
│  │  True Holding Pen (deep copy):      │    │
│  │  vkCmdCopyImage → layer-owned image │    │
│  │  (binary semaphore signals ready)   │    │
│  │           │                         │    │
│  │  Return XR_SUCCESS immediately ◄────┘    │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌─ Runtime Thread ────────────────────┐    │
│  │  xrWaitFrame / xrBeginFrame         │    │
│  │           │                         │    │
│  │  Decision: on-time or deadline miss?│    │
│  │     │                    │          │    │
│  │     ▼                    ▼          │    │
│  │  Path A:             Path B:        │    │
│  │  Submit fresh        Synthesize:    │    │
│  │  held frame          pre-warp →     │    │
│  │                      OFA → stereo   │    │
│  │                      → synthesis    │    │
│  │                      → hole fill    │    │
│  │           │                         │    │
│  │           ▼                         │    │
│  │  xrEndFrame (layer-owned images)    │    │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
        │
        ▼  xrEndFrame (layer-owned color/depth)
OpenXR Runtime / Compositor (PimaxXR / SteamVR)
```

**LSR Fallback** (parallel path): if app misses deadline, skip synthesis and apply pose-only reprojection to last good frame before submission.

---

## System 1: Frame Acquisition & 6DoF Data

**Trigger:** Each `xrEndFrame` call.

**Captured data:**
- `color[L]`, `color[R]` — VkImage handles for current rendered frames (from active swapchain image)
- `depth[L]`, `depth[R]` — VkImage handles for depth (if available)
- `pose_render` — XrPosef at the time the frame was rendered (from `xrLocateViews` at render time)
- `pose_display` — XrPosef at projected display time (from `xrEndFrame` layer projection data)
- `predicted_display_time` — XrTime for the upcoming display
- `frame_index` — monotonic counter

**Depth source:** `XR_KHR_composition_layer_depth` is the preferred path. The layer checks if the app submits `XrCompositionLayerDepthInfoKHR` alongside its projection layers. **Depth is required for full-quality synthesis.** If Star Citizen does not submit depth via this extension (expected), an alternative acquisition path must be researched — options include NVAPI depth readback, game-specific hooks, or a per-pixel depth estimation pass from motion parallax. This is a critical early research item.

**Pose delta:** `delta = pose_display * inverse(pose_render)` — this quaternion/translation delta is the core input to pre-warp (System 2) and LSR (System 7).

---

## System 2: OFA Motion Vectors

**Purpose:** Generate a dense per-pixel motion vector field between frame N−1 and frame N, representing scene-relative motion only (head motion removed by pre-warp).

### Pre-warp (pose delta correction)
Before feeding frames to OFA, apply the pose delta as a homographic warp to frame N−1. This reprojects the previous frame to the camera position/orientation of frame N, removing the dominant global shift caused by head movement. The OFA then sees only object motion and parallax — dramatically improving vector quality.

### Vulkan/CUDA interop
OFA runs in CUDA. Vulkan images must be shared:
1. Allocate VkImage with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`
2. Export memory handle via `VK_KHR_external_memory_win32`
3. Import into CUDA as `CUexternalMemory` → map to `CUarray`
4. Use `VkSemaphore` (exported) ↔ `CUexternalSemaphore` for GPU synchronization

This pattern is applied to both input frames and to the output motion vector buffer.

### NVOf SDK 5.0.7
```
NvOFCreateInstanceCuda()     → create OFA instance
NvOFInit()                   → configure: resolution, output grid size (4×4 recommended)
NvOFExecute()                → submit frame pair → motion vectors
```

### Stereo handling
OFA runs on the **left eye only**. Left-eye motion vectors are then adapted to the right eye by applying an IPD offset transform. The exact transform (horizontal shift + small perspective correction) will be validated empirically. This halves OFA compute cost while exploiting the high correlation between L/R eye motion fields.

**Output:** Dense 2D motion vector field, one entry per 4×4 px block of the left eye image. Right-eye vectors derived from this.

---

## System 3: Bidirectional Frame Warping

**Purpose:** Synthesize a new frame at time T+0.5 between frame N (at T) and frame N+1 (at T+1).

### Forward warp
Warp frame N toward T+0.5: for each pixel at position P with motion vector V, write pixel to P + 0.5·V in the output. This is a scatter operation (forward splatting).

### Backward warp
Warp frame N+1 toward T+0.5: for each pixel at position P in the output, sample frame N+1 at P + 0.5·V_inverted. This is a gather operation.

### Blend
Combine forward and backward warps weighted by confidence:
- **Vector consistency:** pixels where forward and backward vectors agree → high confidence
- **Occlusion map:** regions where one warp produces no coverage → use the other
- **Depth-guided conflict resolution:** where both warps produce coverage, the pixel with smaller depth value (closer surface) wins

### Output
- Synthesized frame VkImage (per eye)
- Hole map: bitmask of pixels where neither warp produced reliable coverage (fast-moving object edges, disocclusion regions)

---

## System 4: Hole Filling

**Purpose:** Fill pixels marked in the hole map that the warp could not reliably synthesize.

**Algorithm (current):** Hierarchical push-pull — a two-pass mipmap-based approach. The push phase downsamples the frame into a mipmap chain, ignoring pixels flagged in the hole map. The pull phase traverses back up the chain, blending lower-resolution valid colors exclusively into the hole pixels. This produces smooth, artifact-free fills with no sharp boundary discontinuities.

**Interface:**
```
fill(VkImage frame, VkImage hole_map) -> VkImage filled_frame
```

**Design constraint:** This interface is intentionally stable. The push-pull fill implementation can be replaced with an AI inpainting model (e.g., a small U-Net or diffusion inpainting) without modifying callers. The hole map drives both implementations identically.

---

## System 5: Dynamic Frame Rate Targeting

**Purpose:** Inject the correct number of synthetic frames to hit target display rate regardless of app native FPS.

**Target rate:** Native panel refresh (90Hz for Pimax Dream Air). Configurable.

**Bypass mode:** If app native FPS >= target rate, the synthesis pipeline is skipped entirely. Zero overhead in this case.

**Injection ratios:**
| App FPS | Target FPS | Synthetics per real frame |
|---|---|---|
| 45 | 90 | 1 |
| 30 | 90 | 2 |
| 60 | 90 | 0 or 1 (alternating) |
| 72 | 90 | 0 or 1 (staggered 1-in-4) |

**Non-integer ratios** require a frame pacing buffer: a ring buffer of `(display_timestamp, frame_type: real|synthetic)` entries. The pacing logic determines when to inject a synthetic frame based on timestamp gaps, avoiding judder at fractional ratios.

**Frame timing measurement:** Track `xrEndFrame` call intervals (rolling average over last 8 frames) to estimate app native FPS.

---

## System 6: Foveated Processing

**Purpose:** Reduce synthesis compute cost by applying full-quality processing only within the foveal region.

**Eye tracking source:** `XR_EXT_eye_gaze_interaction` is the standard extension. Pimax Dream Air may expose a proprietary extension for lower-latency gaze data — confirm at integration time.

**Processing zones:**
- **Foveal region** (within ~15° of gaze point): full-resolution OFA vectors, full-quality warp and hole fill
- **Peripheral region** (outside foveal radius): OFA runs at half resolution, hole fill uses nearest-neighbor only

**Gaze latency:** Use the most recently available gaze sample. If eye tracking data is unavailable or stale (>8ms), fall back to a fixed center-screen radius that covers the full foveal zone.

---

## System 7: LSR Fallback (Late-Stage Reprojection)

**Purpose:** Prevent judder when the app misses its `xrEndFrame` deadline.

**Trigger:** App has not submitted a new frame within the expected interval, or synthesis pipeline output is not ready in time.

**Action:**
1. Take the last successfully synthesized (or real) frame
2. Compute pose delta: `delta = current_display_pose * inverse(last_frame_pose)`
3. Apply delta as a reprojection warp to the last frame
4. Submit the reprojected frame in place of a synthesized frame

**Warp type:** Rotation-only warp if depth is unavailable (correct for rotational head motion, produces parallax error on translational motion). Full 6DoF warp if depth is available (corrects both rotation and translation).

This is the safety net that prevents judder from propagating to the user. It shares the pose data pipeline from System 1 and the warp infrastructure from System 3.

---

## Frame Submission & Decoupled Runtime Thread

**EAC-Safe Architecture:** The layer operates entirely within the OpenXR API boundary. There is **no Vulkan API interception** (no `vkCreateDevice` hook, no `vkCmdPipelineBarrier` injection). This ensures compatibility with Easy Anti-Cheat and avoids Vulkan companion layer complexity.

### True Holding Pen (Layer-Owned Images)

The layer allocates a private ring buffer of Vulkan color images (and later depth images) that it fully owns. When the app calls `xrEndFrame`, the layer:
1. Issues `vkCmdCopyImage` (or a compute-shader copy) from the app's swapchain image into the next available layer-owned image.
2. Records the frame's predicted display time and pose metadata alongside the copied image.
3. Returns `XR_SUCCESS` to the app immediately — the app thread is never throttled by compositor pacing.

Only layer-owned images ever cross the app-thread/runtime-thread boundary. The app's Vulkan memory is never touched by the runtime thread.

### Thread Synchronization: Binary Semaphores + Fences

Timeline semaphores are **not used**. Thread-crossing synchronization relies exclusively on:
- **Binary `VkSemaphore`** — for GPU-side ordering between the copy command and the runtime-thread submission.
- **`VkFence`** — for CPU-side readiness checks before re-recording or re-using a ring slot.

This keeps the synchronization model simple, maximally compatible, and auditable.

### Decoupled Runtime Thread

A dedicated thread owns the full `xrWaitFrame -> xrBeginFrame -> xrEndFrame` loop to the compositor:
- **Path A (on-time):** Submit the freshest layer-owned copied frame.
- **Path B (deadline miss):** Synthesize from the layer-owned buffer using fractional Δt motion scaling and submit the synthesized result.

The runtime thread never accesses application memory. Shutdown calls `vkQueueWaitIdle` or `vkDeviceWaitIdle` to drain in-flight work before freeing layer-owned images.

### Phase 3 Color-Only Constraint

During Phase 3 stabilization, all `XrCompositionLayerDepthInfoKHR` chains are stripped before submission. Depth is re-introduced in Phase 4 after the binary sync and deep-copy logic is proven correct in color-only mode.

---

---

## Key Open Questions / Research Items

| Question | Impact | Status |
|---|---|---|
| Depth source for Star Citizen | Required for full-quality warp (Phase 4) | Research needed |
| OFA latency on RTX 5070 Ti | Must fit ~4-5ms of ~11ms budget | Profile on hardware |
| Eye tracking OpenXR extension for Pimax Dream Air | Foveated processing | Confirm at integration |
| IPD-based stereo vector adaptation accuracy | Right-eye synthesis quality | Validate empirically |
| Binary semaphore ring sizing (color holding pen) | Frame pacing stability | Tune at Phase 3 integration |
| Depth image format compatibility with compositor | Phase 4 depth chain reintegration | Validate at Phase 4 |
