# Architecture: OpenXR Motion Smoothing Layer

## Pipeline Overview (Current Direction)

```
App (Star Citizen / Vulkan)
        │
        ▼  xrEndFrame(frameEndInfo)
┌─────────────────────────────────────────────┐
│           OpenXR API Layer (this project)   │
│                                             │
│  ┌─ Frame Capture ──────────────────────┐   │
│  │  color[L,R], depth[L,R],             │   │
│  │  pose_render, pose_display, timing   │   │
│  └──────────────────────────────────────┘   │
│          │                                  │
│          ▼                                  │
│  ┌─ 6DoF Pre-warp ──────────────────────┐   │
│  │  Apply pose delta to frame N         │   │
│  │  Remove global head-motion before    │   │
│  │  OFA — isolates scene-relative flow  │   │
│  └──────────────────────────────────────┘   │
│          │                                  │
│          ▼                                  │
│  ┌─ OFA Motion Vectors (left eye) ──────┐   │
│  │  Vulkan→CUDA interop                 │   │
│  │  NVOf SDK 5.0.7                      │   │
│  │  Output: dense vector grid (4×4 px)  │   │
│  └──────────────────────────────────────┘   │
│          │                                  │
│          ▼                                  │
│  ┌─ Stereo Vector Adaptation ───────────┐   │
│  │  Project left-eye vectors to right   │   │
│  │  eye via IPD offset                  │   │
│  └──────────────────────────────────────┘   │
│          │                                  │
│          ▼                                  │
│  ┌─ Bidirectional Frame Warp ───────────┐   │
│  │  Forward warp: frame N → T+0.5       │   │
│  │  Backward warp: frame N+1 → T+0.5    │   │
│  │  Depth-guided blend + occlusion map  │   │
│  │  Output: synth frame + hole map      │   │
│  └──────────────────────────────────────┘   │
│          │                                  │
│          ▼                                  │
│  ┌─ Hole Filling ───────────────────────┐   │
│  │  Math inpainting on hole map pixels  │   │
│  │  (swappable AI slot)                 │   │
│  └──────────────────────────────────────┘   │
│          │                                  │
│  ┌─ Queue Isolation + Holding Pen ──────┐   │
│  │  App stays on Queue 0                │   │
│  │  Runtime/compositor moved to Queue 1 │   │
│  │  Deep-copy app frame to layer image  │   │
│  └──────────────────────────────────────┘   │
│          │                                  │
│          ▼                                  │
│  ┌─ Decoupled Runtime Thread ───────────┐   │
│  │  Owns xrWaitFrame/Begin/End cadence  │   │
│  │  Path A: submit queued real frame    │   │
│  │  Path B: synth fallback on deadline  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
        │
        ▼  xrEndFrame (modified layer info)
OpenXR Runtime / Compositor (PimaxXR)
```

**Safety fallback:** If capability gates fail (no controlled Vulkan path/queue redirect), the layer forces passthrough and does not start decoupled submission.

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

## System 5: Decoupled Runtime Pacing (Color-Only First)

**Purpose:** Keep runtime submission at headset cadence while the app thread runs unthrottled.

**Target rate:** Native panel refresh (90Hz for Pimax Dream Air). Configurable.

**Phase 3A behavior (color-only):** No depth chains, no mandatory synthesis. Submit newest queued real frame when available.

**Phase 3B behavior:** If deadline is missed and no fresh real frame exists, synthesize from recent layer-owned color frames.

**Holding pen:** Bounded ring buffer of layer-owned copied color images + timestamps. Never pass raw app-owned image pointers across thread boundary.

**Fractional Δt scaling:** motion vectors and pose warp must be scaled to runtime target `predictedDisplayTime` using actual buffered frame timestamps.

**Queue model:** app rendering remains on Queue 0; runtime/compositor submission uses Queue 1 after OpenXR negotiation/session interception.

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

## Frame Submission

**Purpose:** Deliver real or synthesized layer-owned frames to the downstream runtime from the decoupled runtime thread.

**Mechanism:** App thread captures and deep-copies frame data; runtime thread performs the actual downstream `xrEndFrame` calls at runtime cadence with layer-owned images.

**Synthetic frame swapchain:** A separate VkSwapchainKHR (or XrSwapchain) must be created by the layer to hold synthetic frame output. This is allocated at session creation time with appropriate usage flags for compute write + composition source.

**Complexity notes:**
- Swapchain images must be in `XR_SWAPCHAIN_IMAGE_LAYOUT_COLOR_OPTIMAL` before submission.
- Runtime thread must preserve strict frame order (`xrWaitFrame -> xrBeginFrame -> xrEndFrame`).
- Queue isolation is mandatory before asynchronous submission; no Vulkan submit/present detours.
- On `xrDestroySession`, stop runtime thread, idle queue/device, then free layer-owned images.

---

## Key Open Questions / Research Items

| Question | Impact | Status |
|---|---|---|
| Depth source for Star Citizen | Required for full-quality warp | Research needed |
| OFA latency on RTX 5070 Ti | Must fit ~4-5ms of ~11ms budget | Profile on hardware |
| Eye tracking OpenXR extension for Pimax Dream Air | Foveated processing | Confirm at integration |
| IPD-based stereo vector adaptation accuracy | Right-eye synthesis quality | Validate empirically |
| Queue-index redirect compatibility across runtimes | Decoupled submission stability | Validate on SteamVR/PimaxXR |
| Layer-owned image lifetime + teardown ordering | Avoid vkFreeMemory-in-use failures | Validate in shutdown stress |
