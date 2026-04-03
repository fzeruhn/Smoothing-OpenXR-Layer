# OpenXR Motion Smoothing Layer

A custom [OpenXR API layer](https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#api-layers) that implements high-performance motion smoothing for VR. It sits between the OpenXR runtime and the application, intercepting `xrEndFrame` to synthesize intermediate frames before they reach the compositor. The goal is to meaningfully surpass ASW/SteamVR motion smoothing by fully exploiting Blackwell OFA, depth-guided warping, and foveated processing.

**Hardware targets:**
- HMD: Pimax Dream Air (4K per eye, 90Hz)
- GPU: NVIDIA RTX 5070 Ti (Blackwell architecture)

**Primary validation target:** Star Citizen on Vulkan

**This project is Vulkan-only.** There are no plans for D3D11/D3D12 support.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full system design and [`ROADMAP.md`](ROADMAP.md) for the build sequence.

---

## Implementation Status

| Roadmap Item | Status |
|---|---|
| 1. Vulkan/CUDA Interop Foundation | ✅ Complete |
| 1.5. OpenXR Data Pipeline | 🔄 Partial |
| 2. 6DoF Pose Data Pipeline | ⏳ Pending |
| 3. OFA Integration (NVOf SDK 5.0.7) | ✅ Complete |
| 4. Pre-OFA Pose Pre-warp | ⏳ Pending |
| 5. Depth Acquisition | ⏳ Pending |
| 6. LSR Fallback | ⏳ Pending |
| 7. Bidirectional Frame Synthesis | ✅ Complete |
| 8. Stereo Vector Adaptation | ⏳ Pending |
| 9. Hole Filling (Push-Pull) | ✅ Complete |
| 10. Frame Submission | ⏳ Pending |
| 11. Dynamic Frame Rate Targeting | ⏳ Pending |
| 12. Foveated Processing | ⏳ Pending |

---

## Prerequisites

| Tool | Version |
|---|---|
| Visual Studio | 2026 |
| CUDA Toolkit | 13.2 |
| NVOf SDK | 5.0.7 |
| Windows SDK | 10.0 |
| Target OS | Windows 11 |

- **Visual Studio 2026** — C++ desktop workload + NuGet package manager
- **Python 3** — in PATH (required for OpenXR header generation during build)
- **CUDA 13.2 Toolkit** — install from NVIDIA (not via NuGet)
- **NVOf SDK 5.0.7** — install from NVIDIA Optical Flow SDK

NuGet dependencies (`fmt`, `OpenXR.Headers`, `OpenXR.Loader`, `WIL`) restore automatically at build time.

---

## Building

> **First-time setup:** Run `git submodule update --init --recursive` before building. The pre-build code generation step requires the OpenXR submodules to be initialized.

```powershell
# Debug build (always restore NuGet packages first)
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m"

# Release build
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Release /p:Platform=x64 /m"

# Clean
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /t:Clean /p:Configuration=Debug /p:Platform=x64"
```

Primary output: `bin\x64\Debug\` or `bin\x64\Release\`.

### Runtime DLLs

After the first build, copy the CUDA runtime DLL to the output directory — it is **not** in the default PATH:
```
copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\cudart64_13.dll" bin\x64\Debug\
copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\cudart64_13.dll" bin\x64\Release\
```

> Do **not** use `cudart_static.lib` in projects built with `/MDd` — the CRT mismatch (`/MT` vs `/MDd`) causes a pre-`main()` crash with no output.

`nvofapi64.dll` ships with the NVIDIA driver and lives in `C:\Windows\System32\` — no manual copying needed.

---

## Standalone Tests

Four headless test executables validate the completed pipeline subsystems. Run them from the solution root:

| Test | What it validates |
|---|---|
| `bin\x64\Debug\interop-test.exe` | Vulkan/CUDA round-trip: CUDA kernel writes pattern, Vulkan reads back |
| `bin\x64\Debug\ofa-test.exe` | OFA optical flow end-to-end: random-noise frames with a known pixel shift |
| `bin\x64\Debug\synthesis-test.exe` | Bidirectional frame synthesis: checkerboard translation |
| `bin\x64\Debug\hole-fill-test.exe` | Push-pull hole filling: solid and gradient frames with synthetic holes |

`ofa-test.exe` also writes `ofa-test-output.png` (color-coded flow field, red=X, green=Y, ±16px range) to the current directory.

---

## Install / Uninstall

```powershell
# Register the layer with the OpenXR loader (run as admin or from elevated PS)
.\scripts\Install-Layer.ps1

# Uninstall
.\scripts\Uninstall-Layer.ps1
```

Layer descriptor JSONs: `openxr-api-layer/openxr-api-layer.json` (x64) and `openxr-api-layer/openxr-api-layer-32.json` (x86).

---

## Tracing

Use `scripts/Tracing.wprp` with Windows Performance Recorder (WPR) to capture ETW traces. The layer emits TraceLogging events via `framework/log.h` macros.

---

## Project Structure

```
openxr-api-layer/
  layer.cpp                    Main layer logic — OpenXR hooks live here
  layer.h                      Layer class definition
  vulkan_cuda_interop.h/.cpp   SharedImage + SharedSemaphore RAII wrappers (Item 1)
  frame_synthesizer.h/.cu      FrameSynthesizer RAII + CUDA kernels (Item 7)
  hole_filler.h/.cu            HoleFiller push-pull CUDA kernel (Item 9)
  pch.h / pch.cpp              Precompiled headers
  framework/
    dispatch.h/.cpp            OpenXR function dispatch (template framework)
    entry.cpp                  DLL entry + loader negotiation (template framework)
    log.h/.cpp                 TraceLogging provider + macros
    util.h                     Formatting utilities (poses, FOV, vectors)
  utils/
    general.h/.cpp             General utilities
    inputs.h / input.cpp       Input handling utilities
    composition.cpp            Composition layer utilities
interop-test/
  main.cpp                     Headless Vulkan + CUDA round-trip test harness
  fill_pattern.cu              CUDA kernel + C-callable launcher
ofa-test/
  main.cpp                     OFA end-to-end test: random noise frames, shift validation, PNG output
synthesis-test/
  main.cpp                     Bidirectional synthesis end-to-end test
hole-fill-test/
  main.cpp                     Push-pull hole fill end-to-end test
external/
  OpenXR-SDK/                  Khronos OpenXR SDK (submodule)
  OpenXR-SDK-Source/           Khronos OpenXR SDK source (submodule)
  OpenXR-MixedReality/         Microsoft OpenXR MixedReality (submodule)
  PVR/                         Pimax SDK (eye tracking)
scripts/
  Install-Layer.ps1            Registers the layer in the Windows registry
  Uninstall-Layer.ps1          Unregisters the layer
  Tracing.wprp                 WPR trace capture profile
```

---

DISCLAIMER: This software is distributed as-is, without any warranties or conditions of any kind. Use at your own risks.
