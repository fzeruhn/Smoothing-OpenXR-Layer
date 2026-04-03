# Development Roadmap

Each item is listed in dependency order. Complete earlier items before starting later ones — most items depend on shared infrastructure from the items above them.

---

## 1. Vulkan/CUDA Interop Foundation ✅ COMPLETE

**Goal:** Prove that a VkImage can be shared with CUDA, written to by a compute shader or CUDA kernel, and read back correctly.

**Work:**
- Create VkImage with external memory flags (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`)
- Export memory handle via `VK_KHR_external_memory_win32`
- Import into CUDA as `CUexternalMemory`, map to `CUarray`
- Create/export `VkSemaphore` for GPU sync, import as `CUexternalSemaphore`
- Write a test: CUDA kernel fills image with a pattern, verify with Vulkan readback

**Delivered:** `openxr-api-layer/vulkan_cuda_interop.h/.cpp` — `SharedImage` and `SharedSemaphore` RAII wrappers. `interop-test/` — standalone test exe confirming full round-trip on RTX 5070 Ti. `[PASS] Vulkan/CUDA interop verified (256x256 RGBA, pattern round-trip)`.

**Why first:** Every subsequent system (OFA, warp, fill) depends on this interop pattern. Validate it in isolation before building anything on top of it.

---

## 1.5. OpenXR Data Pipeline - Partially complete

**Goal:** Get all informarion that will be needed for later calculations from openxr, and do any nessecary calculations to make it usable

**Work:**
- Hook OpenXR (xrLocateViews) to extract the headset's physical FOV (parsing XrFovf to account for asymmetric/canted displays).
- TODO: fill out

---

## 2. 6DoF Pose Data Pipeline

**Goal:** Reliably capture render-time pose and display-time pose at each `xrEndFrame`, and make them available to downstream systems.

**Work:**
- Extract `XrPosef` from `xrEndFrame` projection layer data (display-time pose)
- Hook `xrLocateViews` to capture render-time pose at frame submission, or store the pose from the most recent `xrWaitFrame`/`xrBeginFrame` pair
- Define `PoseData` struct: `{ render_pose, display_pose, predicted_display_time, frame_index }`
- Compute `pose_delta = display_pose * inverse(render_pose)` — this is the input to pre-warp and LSR
- Store per-frame pose history (ring buffer, at least 2 frames deep)

**Why second:** Pre-warp (item 4), LSR (item 6), and frame submission timing all depend on this. Design the struct now so downstream systems can be written against it.

---

## 3. OFA Integration ✅ COMPLETE

**Goal:** Feed two VkImages to NVIDIA Optical Flow and get back a motion vector field.

**Work:**
- Add NVOf SDK 5.0.7 to build system (include/lib paths, DLL deployment)
- Initialize `NvOFCreateInstanceCuda()` at session creation
- Wire up left-eye frame pair (previous frame, current frame) → OFA via CUDA surfaces from item 1
- Configure: resolution = left eye image resolution, grid size = 4×4 px
- Execute `NvOFExecute()`, retrieve motion vector output buffer
- Validate output visually (render motion vector field as color image, check it makes sense on test content)

**Why third:** OFA is the backbone of the entire synthesis pipeline. Validate on its own before layering pre-warp on top.

---

## 4. Pre-OFA Pose Pre-warp - Can be done headless

**Goal:** Apply pose delta to frame N−1 before feeding to OFA, so that OFA sees only scene-relative motion.

**Work:**
- Compute homography matrix from `pose_delta` (from item 2) and camera intrinsics (FOV from `xrEndFrame`)
- Apply homography warp to previous frame as a CUDA kernel (to keep memory in the current execution context) before OFA execution
- Compare OFA vector field quality with and without pre-warp (smoother, smaller magnitude in static scenes)

**Why fourth:** Improves OFA quality across the whole pipeline. Build on top of validated OFA from item 3.

---

## 5. Depth Acquisition

**Goal:** Obtain per-pixel depth data for both eyes at every frame.

**Primary path:** Check if the app submits `XrCompositionLayerDepthInfoKHR` alongside its projection layers in `xrEndFrame`. Parse depth VkImage from this extension.

**Fallback research (if primary path unavailable in Star Citizen):**
- Option A: NVAPI / NvIFR depth readback from app's render pipeline
- Option B: Depth estimation from consecutive frames using motion parallax (compute-based, rough but useful)
- Option C: Game-specific injection hook (last resort, fragile)

**Deliverable:** Working depth acquisition for at least one path. Document which path is in use and its limitations.

**Why fifth:** Depth-guided warping (item 7) and LSR 6DoF correction (item 6) both require depth. Establish the source before building the consumers.

---

## 6. LSR Fallback (Late-Stage Reprojection)

**Goal:** When the app misses its frame deadline, reproject the last good frame using the current pose delta rather than dropping or juddering.

**Work:**
- Detect frame deadline miss (app `xrEndFrame` interval exceeds 1.2× expected interval)
- Take last good synthesized/real frame VkImage
- Apply pose delta as reprojection warp: rotation-only if no depth, full 6DoF if depth available (item 5)
- Submit reprojected frame via downstream `xrEndFrame` with correct `predictedDisplayTime`

**Why sixth:** Validates pose pipeline (item 2) and warp primitives in a simpler context than full synthesis. Also useful standalone — can ship this as a safety net even before full synthesis is ready.

---

## 7. Bidirectional Frame Synthesis ✅ COMPLETE

**Goal:** Given frame N, frame N+1, motion vectors, and depth, synthesize a frame at T+0.5.

**Work:**
- Forward warp: scatter frame N pixels using motion vectors (P → P + 0.5·V)
- Backward warp: gather from frame N+1 using reversed vectors
- Confidence map: per-pixel weight based on vector consistency and coverage
- Depth-guided blend: in conflict pixels, closer surface wins
- Output hole map: bitmask where neither warp produced reliable coverage

**Validation:** Synthesize frames offline (fixed test images) before integrating into the live pipeline.

---

## 8. Stereo Vector Adaptation - Can be done headless

**Goal:** Derive right-eye motion vectors from left-eye OFA output.

**Work:**
- Derive focal length (f) from the acquired horizontal FOV
- Calculate binocular disparity per pixel using the depth map
- Apply calculated disparity offset to shift left-eye vectors to the right eye
- Apply perspective correction and handle disocclusion/overlap based on depth
- Validate: compare adapted right-eye vectors to a direct right-eye OFA run (run OFA on both eyes in test mode, measure error)
- Tune transform until adapted vectors are within acceptable error margin

---

## 9. Hole Filling - Can be done headless

**Goal:** Fill pixels flagged in the hole map from item 7 without introducing sharp, distracting artifacts.

**Work:**
- Implement a Hierarchical Push-Pull CUDA kernel:
- Push (Downsample): Generate a mipmap chain of the frame, ignoring pixels flagged in the hole map.
- Pull (Upsample): Traverse back up the mipmap chain, blending the lower-resolution valid colors exclusively into the hole map pixels.

- Accept `(frame, hole_map)` → `filled_frame` interface (per ARCHITECTURE.md)
- Validate: compare filled output against ground truth on synthetic test cases with known disocclusions

---

## 10. Frame Submission

**Goal:** Inject synthesized frame(s) back into the OpenXR compositor by modifying the downstream `xrEndFrame` call.

**Work:**
- At session creation, allocate a synthetic output XrSwapchain with appropriate usage flags (compute write + color attachment + composition source)
- At each synthesis cycle, acquire a synthetic swapchain image, write synthesized output to it, release it
- Build modified `XrFrameEndInfo`: replace projection layer swapchain reference with synthetic swapchain
- For 2× synthesis (45→90): call downstream `xrEndFrame` twice — once with real frame (T), once with synthetic frame (T+0.5) — each with correct `predictedDisplayTime`
- Handle image layout transitions to `XR_SWAPCHAIN_IMAGE_LAYOUT_COLOR_OPTIMAL` before submission

**Why tenth:** Frame submission requires all upstream pipeline stages to produce valid output. Integrate last, after synthesis quality is validated offline.

---

## 11. Dynamic Frame Rate Targeting

**Goal:** Automatically determine injection count based on live app frame timing, with zero-overhead bypass when app hits target rate.

**Work:**
- Frame timing tracker: rolling average of `xrEndFrame` call intervals over last 8 frames
- Derive native app FPS estimate
- If native FPS >= target (90Hz): bypass synthesis, pass through real frames unchanged
- Otherwise: compute injection count; implement non-integer ratio handling via frame pacing buffer (ring buffer of `(timestamp, frame_type)`)
- Validate: test at 45, 60, 72, 30 fps app rates

---

## 12. Foveated Processing

**Goal:** Use eye tracking gaze data to reduce OFA and warp resolution in the peripheral region, reducing GPU cost.

**Work:**
- Query `XR_EXT_eye_gaze_interaction` (or Pimax-specific extension — confirm at integration time)
- Define foveal radius (start with ~15° of visual angle)
- Gate OFA grid resolution: full resolution inside foveal radius, half resolution outside
- Gate hole fill quality: edge-directed fill inside foveal zone, nearest-neighbor outside
- Implement gaze latency fallback: if tracking unavailable or stale (>8ms), use fixed center radius

---

## Future (Not Sequenced)

- **AI inpainting slot:** Replace math hole fill (item 9) with a small inpainting model. Interface is stable — no pipeline restructure needed.
- **Star Citizen depth extraction:** If the preferred paths (item 5) prove insufficient, invest in game-specific depth acquisition.
- **Per-frame quality metrics:** Automated SSIM / LPIPS comparison between synthesized frames and ground-truth interpolations for regression testing.
- **Auto IPD getting:** Get IPD from headset API / SDK to automatically do the stero vector adaptation without user input IPD