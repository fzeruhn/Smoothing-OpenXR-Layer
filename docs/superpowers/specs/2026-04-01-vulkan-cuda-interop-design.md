# Design: Vulkan/CUDA Interop Foundation

**Date:** 2026-04-01
**Roadmap item:** 1 — Vulkan/CUDA Interop Foundation
**Status:** Approved, pending implementation

---

## Goal

Prove that a `VkImage` can be shared with CUDA, written to by a CUDA kernel, and read back correctly via Vulkan. This is the foundational primitive every downstream system (OFA, pre-warp, hole fill, frame submission) depends on.

---

## Decisions

| Question | Decision |
|---|---|
| Validation strategy | Standalone test executable (not a startup self-test or deferred validation) |
| Code organization | New dedicated `vulkan_cuda_interop` module shared between layer and test |
| API shape | RAII wrapper structs (`SharedImage`, `SharedSemaphore`) |

---

## Section 1: `vulkan_cuda_interop` Module

**Files:** `openxr-api-layer/vulkan_cuda_interop.h`, `openxr-api-layer/vulkan_cuda_interop.cpp`

### `SharedImage`

Owns the full lifetime of a Vulkan/CUDA shared image resource:

- **Vulkan side:** `VkImage` + `VkDeviceMemory` allocated with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`. Memory exported via `VK_KHR_external_memory_win32`.
- **CUDA side:** `CUexternalMemory` imported from the Win32 handle, mapped to a `CUmipmappedArray`, with a `CUarray` level-0 accessor.
- **Constructor:** takes `VkDevice`, `VkPhysicalDevice`, width, height, `VkFormat`. Allocates and wires both sides.
- **Destructor:** releases CUDA resources first (`cuDestroyExternalMemory`), then Vulkan (`vkDestroyImage`, `vkFreeMemory`). Order matters — CUDA must release before Vulkan frees the underlying memory.
- **Accessors:** `vkImage() -> VkImage`, `cuArray() -> CUarray`
- **Move-only:** deleted copy constructor and copy assignment.

### `SharedSemaphore`

Owns the full lifetime of a Vulkan/CUDA shared semaphore:

- **Vulkan side:** `VkSemaphore` created with `VkExportSemaphoreCreateInfo`, exported as a Win32 handle via `VK_KHR_external_semaphore_win32`.
- **CUDA side:** `CUexternalSemaphore` imported from the Win32 handle.
- **Constructor:** takes `VkDevice`. Creates and exports.
- **Destructor:** `cuDestroyExternalSemaphore` first, then `vkDestroySemaphore`.
- **Helpers:**
  - `signal(CUstream)` — calls `cuSignalExternalSemaphoresAsync` (CUDA signals Vulkan)
  - `wait(CUstream)` — calls `cuWaitExternalSemaphoresAsync` (CUDA waits on Vulkan)
  - `vkSemaphore() -> VkSemaphore` — for use in `VkSubmitInfo::pWaitSemaphores` / `pSignalSemaphores`
- **Move-only.**

### API Contract

Both types are constructed in a valid-or-throws manner — if Vulkan or CUDA setup fails, the constructor throws with a descriptive message. No two-phase init, no `bool isValid()` checks at call sites.

---

## Section 2: Build System Changes

### `openxr-api-layer.vcxproj`

Add to all configurations (Debug/Release × Win32/x64):

- **Additional Include Directories:** `$(CudaToolkitDir)include`
- **Additional Library Directories:** `$(CudaToolkitDir)lib\x64`
- **Additional Dependencies:** `cuda.lib` (driver API — not cudart)
- `vulkan_cuda_interop.cpp` added as a standard `ClCompile` item (driver API is a regular `.cpp`, no `.cu` extension needed)

`$(CudaToolkitDir)` resolves from the CUDA 13.2 Toolkit install. The CUDA MSBuild integration sets this automatically when the CUDA Toolkit is installed.

### New Test Project: `interop-test`

**Location:** `interop-test/interop-test.vcxproj`

- **Type:** Console application (`.exe`)
- **Platform:** x64 only (matches GPU target)
- **Configurations:** Debug and Release
- **Dependencies:**
  - Vulkan headers and loader (same NuGet packages as the layer, or via `$(VULKAN_SDK)`)
  - `cuda.lib` + `$(CudaToolkitDir)include`
  - CUDA runtime (`cudart.lib`) for kernel launch from the `.cu` file
  - The CUDA MSBuild extension (`.targets` import) to compile `.cu` files
- **Source files:**
  - `interop-test/main.cpp` — test harness, Vulkan setup, verification logic
  - `interop-test/fill_pattern.cu` — CUDA kernel
  - `openxr-api-layer/vulkan_cuda_interop.cpp` — shared via relative path reference (not duplicated)

Add `interop-test/interop-test.vcxproj` to `SMOOTHING-OPENXR-LAYER.sln` as a new project. It does not need to be a build dependency of the layer — it builds independently.

---

## Section 3: Test Flow

The test is a console application that exits 0 on success, 1 on failure.

### Vulkan Setup (headless)

1. Create `VkInstance` with extensions: `VK_KHR_external_memory_capabilities`, `VK_KHR_external_semaphore_capabilities`
2. Select first discrete GPU (`VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`)
3. Create `VkDevice` with a single compute queue and extensions: `VK_KHR_external_memory`, `VK_KHR_external_memory_win32`, `VK_KHR_external_semaphore`, `VK_KHR_external_semaphore_win32`
4. Create a `VkCommandPool` + `VkCommandBuffer` for the readback step

### Test Execution

1. **Construct** `SharedImage(device, physDevice, 256, 256, VK_FORMAT_R8G8B8A8_UNORM)` and `SharedSemaphore(device)`
2. **CUDA write:** Launch `fill_pattern` kernel on the `CUarray` — each pixel value = `(x + y * 256) % 256` in the R channel, 0 in G/B/A
3. **Sync CUDA→Vulkan:** Call `semaphore.signal(stream)` then `cuStreamSynchronize`
4. **Vulkan readback:**
   - Allocate a host-visible `VkBuffer` (256 × 256 × 4 bytes)
   - Record a command buffer: image layout transition (`UNDEFINED → TRANSFER_SRC_OPTIMAL`), `vkCmdCopyImageToBuffer`, transition back
   - Submit with `pWaitSemaphores = [semaphore.vkSemaphore()]`, `waitDstStageMask = TRANSFER`
   - `vkQueueWaitIdle`
5. **Verify:** Map the buffer, check every pixel's R channel matches `(x + y * 256) % 256`
6. **Report:** Print `[PASS] Vulkan/CUDA interop verified` or `[FAIL] Mismatch at pixel (x, y): expected N, got M`. Exit 0 or 1.

### Cleanup

All RAII destructors run on scope exit. No manual cleanup needed in the happy path.

---

## File Layout After Implementation

```
openxr-api-layer/
  vulkan_cuda_interop.h      NEW — SharedImage, SharedSemaphore declarations
  vulkan_cuda_interop.cpp    NEW — implementation
  layer.cpp                  unchanged (interop not yet wired in)
  openxr-api-layer.vcxproj   MODIFIED — CUDA include/lib paths added
interop-test/
  main.cpp                   NEW — test harness
  fill_pattern.cu             NEW — CUDA pattern kernel
  interop-test.vcxproj       NEW — console app project
SMOOTHING-OPENXR-LAYER.sln   MODIFIED — interop-test project added
```

---

## What This Does NOT Cover

- Wiring `SharedImage`/`SharedSemaphore` into `VulkanFrameProcessor` — that is Roadmap item 3 (OFA integration)
- NvOf SDK integration — Roadmap item 3
- Any frame synthesis logic — downstream items
