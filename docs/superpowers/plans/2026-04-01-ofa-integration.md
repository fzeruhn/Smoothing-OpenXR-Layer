# OFA Integration (Roadmap Item 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `OFAPipeline` — a thin RAII wrapper around the NVOf SDK C API — and validate it end-to-end with a standalone `ofa-test` exe that generates synthetic frames, runs OFA, asserts vector accuracy, and writes a color-coded PNG.

**Architecture:** `OFAPipeline` in its own `ofa_pipeline.h/.cpp` follows the same RAII style as `SharedImage`/`SharedSemaphore`. It loads the NvOf function-pointer table via `NvOFAPICreateInstanceCuda`, manages GPU buffer handles internally, and exposes `loadFrame` / `execute` / `outputData`. The test harness mirrors `interop-test`: a standalone console exe with no Vulkan dependency.

**Tech Stack:** CUDA Driver API (`cuda.lib`), NVOf SDK 5.0.7 C API (`nvOpticalFlowCuda.h`), stb_image_write (header-only PNG), MSVC C++17.

---

## File Map

| Action | Path | Purpose |
|---|---|---|
| Create | `openxr-api-layer/ofa_pipeline.h` | `OFAPipeline` class declaration + error macros |
| Create | `openxr-api-layer/ofa_pipeline.cpp` | Full implementation (not using PCH) |
| Create | `ofa-test/main.cpp` | Test harness: synthetic frames → OFA → validate → PNG |
| Create | `ofa-test/stb_image_write.h` | Drop-in PNG writer |
| Create | `ofa-test/ofa-test.vcxproj` | Standalone console app project |
| Modify | `openxr-api-layer/openxr-api-layer.vcxproj` | Add NvOF include path + register new source files |
| Modify | `SMOOTHING-OPENXR-LAYER.sln` | Add ofa-test project |

---

## Task 1: Build System — Add NvOF includes and new project skeleton

**Files:**
- Modify: `openxr-api-layer/openxr-api-layer.vcxproj`
- Create: `ofa-test/ofa-test.vcxproj`
- Modify: `SMOOTHING-OPENXR-LAYER.sln`

- [ ] **Step 1: Read the existing vcxproj to understand structure**

```bash
# Read the file to see the AdditionalIncludeDirectories lines
# (needed to know exact surrounding XML to target with Edit)
```
Open `openxr-api-layer/openxr-api-layer.vcxproj` and locate all `<AdditionalIncludeDirectories>` elements in `Debug|x64` and `Release|x64` `ItemDefinitionGroup` blocks.

- [ ] **Step 2: Add NvOF include path to Debug|x64 in openxr-api-layer.vcxproj**

Find the Debug|x64 `<AdditionalIncludeDirectories>` line (it currently starts with `$(CUDA_PATH)\include;`). Append the NvOF path before `%(AdditionalIncludeDirectories)`:

```xml
<AdditionalIncludeDirectories>$(CUDA_PATH)\include;C:\VulkanSDK\1.4.341.1\Include;$(ProjectDir);$(ProjectDir)\framework;$(SolutionDir)..\Optical_Flow_SDK_5.0.7\NvOFInterface;$(SolutionDir)\external\OpenXR-SDK\include;$(SolutionDir)\external\OpenXR-SDK\src\common;$(SolutionDir)\external\OpenXR-MixedReality\Shared\XrUtility;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
```

- [ ] **Step 3: Add NvOF include path to Release|x64 in openxr-api-layer.vcxproj**

Same edit for the Release|x64 `ItemDefinitionGroup`. The `<AdditionalIncludeDirectories>` line receives the same `$(SolutionDir)..\Optical_Flow_SDK_5.0.7\NvOFInterface;` insertion.

- [ ] **Step 4: Register ofa_pipeline source files in openxr-api-layer.vcxproj**

Find the `<ItemGroup>` containing `<ClCompile Include="layer.cpp" />` entries and add:

```xml
<ClCompile Include="ofa_pipeline.cpp">
  <PrecompiledHeader>NotUsing</PrecompiledHeader>
</ClCompile>
```

And in the `<ItemGroup>` containing `<ClInclude>` entries:

```xml
<ClInclude Include="ofa_pipeline.h" />
```

- [ ] **Step 5: Generate a GUID for ofa-test**

```powershell
powershell.exe -Command "[System.Guid]::NewGuid().ToString().ToUpper()"
```

Save this GUID — it is needed in the next two steps. Referred to as `$OFA_GUID` below.

- [ ] **Step 6: Create ofa-test/ofa-test.vcxproj**

Read `interop-test/interop-test.vcxproj` first to copy the exact `<PlatformToolset>` value. Then write the new file:

```xml
<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <VCProjectVersion>17.0</VCProjectVersion>
    <ProjectGuid>{$OFA_GUID}</ProjectGuid>
    <RootNamespace>ofa-test</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset><!-- COPY FROM interop-test.vcxproj --></PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset><!-- COPY FROM interop-test.vcxproj --></PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props"
            Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')"
            Label="LocalAppDataProps" />
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props"
            Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')"
            Label="LocalAppDataProps" />
  </ImportGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>Disabled</Optimization>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>_DEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <LanguageStandard>stdcpp17</LanguageStandard>
      <AdditionalIncludeDirectories>$(CudaToolkitDir)include;$(SolutionDir)..\Optical_Flow_SDK_5.0.7\NvOFInterface;$(SolutionDir)openxr-api-layer;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <AdditionalDependencies>cuda.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalLibraryDirectories>$(CudaToolkitDir)lib\x64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
      <OutputFile>$(SolutionDir)bin\$(Platform)\$(Configuration)\$(TargetName)$(TargetExt)</OutputFile>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <LanguageStandard>stdcpp17</LanguageStandard>
      <AdditionalIncludeDirectories>$(CudaToolkitDir)include;$(SolutionDir)..\Optical_Flow_SDK_5.0.7\NvOFInterface;$(SolutionDir)openxr-api-layer;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <AdditionalDependencies>cuda.lib;%(AdditionalDependencies)</AdditionalDependencies>
      <AdditionalLibraryDirectories>$(CudaToolkitDir)lib\x64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
      <OutputFile>$(SolutionDir)bin\$(Platform)\$(Configuration)\$(TargetName)$(TargetExt)</OutputFile>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="main.cpp" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
</Project>
```

Replace both `<!-- COPY FROM interop-test.vcxproj -->` placeholders with the actual toolset string from `interop-test.vcxproj` (e.g. `v143` or `v144`).

- [ ] **Step 7: Add ofa-test to SMOOTHING-OPENXR-LAYER.sln**

Read `SMOOTHING-OPENXR-LAYER.sln`. Find the `interop-test` project entry block and the four configuration lines for it in the `GlobalSection(ProjectConfigurationPlatforms)`. Add analogous entries for ofa-test immediately after:

Project declaration (after the interop-test `EndProject` line):
```
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "ofa-test", "ofa-test\ofa-test.vcxproj", "{$OFA_GUID}"
EndProject
```

Configuration lines (inside `GlobalSection(ProjectConfigurationPlatforms) = postSolution`):
```
		{$OFA_GUID}.Debug|x64.ActiveCfg = Debug|x64
		{$OFA_GUID}.Debug|x64.Build.0 = Debug|x64
		{$OFA_GUID}.Release|x64.ActiveCfg = Release|x64
		{$OFA_GUID}.Release|x64.Build.0 = Release|x64
```

- [ ] **Step 8: Verify solution loads (no build yet)**

```powershell
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /t:ofa-test /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m 2>&1 | Select-String -Pattern 'error|warning' | Select-Object -First 20"
```

Expected: errors about missing `main.cpp` — that's fine. What must NOT appear: MSBuild project load errors or SLN parse errors.

- [ ] **Step 9: Commit**

```bash
git add openxr-api-layer/openxr-api-layer.vcxproj ofa-test/ofa-test.vcxproj SMOOTHING-OPENXR-LAYER.sln
git commit -m "build: add ofa-test project and NvOF include paths"
```

---

## Task 2: Write `ofa_pipeline.h` (interface + macros)

**Files:**
- Create: `openxr-api-layer/ofa_pipeline.h`

- [ ] **Step 1: Create ofa_pipeline.h**

```cpp
#pragma once

// NvOF C API — do not include through pch.h
#include <cuda.h>
#include "nvOpticalFlowCuda.h"
#include "nvOpticalFlowCommon.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error-check macros
// ---------------------------------------------------------------------------

// For the one call that happens before m_hOf exists (NvOFAPICreateInstanceCuda).
#define CHECK_NVOF_INIT(call)                                                     \
    do {                                                                          \
        NV_OF_STATUS _s = (call);                                                 \
        if (_s != NV_OF_SUCCESS)                                                  \
            throw std::runtime_error("NvOFAPICreateInstanceCuda failed: status " +\
                                     std::to_string(static_cast<int>(_s)));       \
    } while (0)

// For all subsequent NvOf calls (m_hOf is valid, error string available).
#define CHECK_NVOF(call)                                                          \
    do {                                                                          \
        NV_OF_STATUS _s = (call);                                                 \
        if (_s != NV_OF_SUCCESS) {                                                \
            char _buf[512] = {};                                                  \
            uint32_t _sz = sizeof(_buf);                                          \
            if (m_hOf) m_api.nvOFGetLastError(m_hOf, _buf, &_sz);                \
            throw std::runtime_error(std::string("NvOF error ") +                 \
                                     std::to_string(static_cast<int>(_s)) +       \
                                     ": " + _buf);                                \
        }                                                                         \
    } while (0)

// For CUDA driver API calls.
#define CHECK_CU(call)                                                            \
    do {                                                                          \
        CUresult _r = (call);                                                     \
        if (_r != CUDA_SUCCESS) {                                                 \
            const char* _s = nullptr;                                             \
            cuGetErrorString(_r, &_s);                                            \
            throw std::runtime_error(std::string("CUDA driver: ") +              \
                                     (_s ? _s : "unknown error"));                \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// OFAPipeline
// ---------------------------------------------------------------------------

class OFAPipeline {
public:
    // Construct and fully initialise the OFA instance.
    // ctx         : active CUDA context (must remain valid for the lifetime of this object)
    // width/height: input frame dimensions in pixels
    // gridSize    : output grid — NV_OF_OUTPUT_VECTOR_GRID_SIZE_4 gives one vector per 4×4 block
    // perfLevel   : quality/speed trade-off
    OFAPipeline(CUcontext ctx,
                uint32_t  width,
                uint32_t  height,
                NV_OF_OUTPUT_VECTOR_GRID_SIZE gridSize  = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4,
                NV_OF_PERF_LEVEL              perfLevel = NV_OF_PERF_LEVEL_MEDIUM);

    ~OFAPipeline();

    OFAPipeline(const OFAPipeline&)            = delete;
    OFAPipeline& operator=(const OFAPipeline&) = delete;

    // Upload an 8-bit grayscale host frame into the named slot.
    //   slot 0 = inputFrame     (current / new frame)
    //   slot 1 = referenceFrame (previous frame)
    //
    // TODO (Item 4 pre-warp hook): replace or augment with
    //   void setInputDevicePtr(int slot, CUdeviceptr devPtr);
    // so the pre-warp kernel can write directly into OFA input buffers.
    void loadFrame(int slot, const void* hostGray8);

    // Submit OFA work asynchronously.
    // `stream` is reserved for the Item-4 pre-warp integration; currently ignored
    // (OFA uses its own internal stream). Caller must call cuCtxSynchronize() before
    // reading outputData().
    void execute(CUstream stream = nullptr);

    // Pointer into the host-side readback buffer.
    // Valid after a cuCtxSynchronize() following execute().
    // Each element is NV_OF_FLOW_VECTOR { int16_t flowx, flowy } in S10.5 fixed-point.
    // Divide by 32.0f to convert to pixel displacement.
    const NV_OF_FLOW_VECTOR* outputData() const { return m_hostOutput.data(); }

    uint32_t outputWidth()  const { return m_outW; }  // width  / gridSize
    uint32_t outputHeight() const { return m_outH; }  // height / gridSize

private:
    void destroy() noexcept;

    NV_OF_CUDA_API_FUNCTION_LIST m_api{};
    NvOFHandle                   m_hOf{};
    NvOFGPUBufferHandle          m_inputBufs[2]{};
    NvOFGPUBufferHandle          m_outputBuf{};

    uint32_t m_width{};
    uint32_t m_height{};
    uint32_t m_outW{};
    uint32_t m_outH{};

    std::vector<NV_OF_FLOW_VECTOR> m_hostOutput;
};
```

- [ ] **Step 2: Verify header compiles in isolation (part of openxr-api-layer build)**

```powershell
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /t:openxr-api-layer /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m 2>&1 | Select-String 'error' | Select-Object -First 20"
```

Expected: build fails with "ofa_pipeline.cpp: No such file" or similar unresolved-symbol errors — NOT header parse errors. If there are include errors (e.g. `nvOpticalFlowCuda.h not found`), fix the include path in the vcxproj before continuing.

- [ ] **Step 3: Commit**

```bash
git add openxr-api-layer/ofa_pipeline.h
git commit -m "feat: add OFAPipeline class declaration and error macros"
```

---

## Task 3: Write ofa-test/main.cpp and drop in stb_image_write.h

**Files:**
- Create: `ofa-test/main.cpp`
- Create: `ofa-test/stb_image_write.h`

- [ ] **Step 1: Download stb_image_write.h**

```powershell
powershell.exe -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h' -OutFile 'ofa-test/stb_image_write.h'"
```

- [ ] **Step 2: Create ofa-test/main.cpp**

```cpp
// ofa-test/main.cpp
// Validates OFAPipeline on synthetic input:
//   Frame 0 (reference): 256x256 checkerboard (32px squares, luma 0/200)
//   Frame 1 (current):   Frame 0 shifted +8px right, +4px down
// Expected output: flow vectors ≈ (+8.0, +4.0) px = (+256, +128) in S10.5.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ofa_pipeline.h"

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

static constexpr int W       = 256;
static constexpr int H       = 256;
static constexpr int SHIFT_X = 8;   // pixels right
static constexpr int SHIFT_Y = 4;   // pixels down

// S10.5 expected: +8px = 8*32 = 256, +4px = 4*32 = 128
static constexpr int16_t EXPECTED_FX = 256;
static constexpr int16_t EXPECTED_FY = 128;
static constexpr int16_t TOLERANCE   = 1;  // ±1 S10.5 unit = ±1/32 px

// Pixels in the central 50x50 region of the 64x64 output are validated.
static constexpr int OUT_W    = W / 4;  // 64
static constexpr int OUT_H    = H / 4;  // 64
static constexpr int ROI_SIZE = 50;
static constexpr int ROI_X0   = (OUT_W - ROI_SIZE) / 2;  // 7
static constexpr int ROI_Y0   = (OUT_H - ROI_SIZE) / 2;  // 7

int main() {
    // -----------------------------------------------------------------------
    // CUDA init
    // -----------------------------------------------------------------------
    CUresult cuErr = cuInit(0);
    if (cuErr != CUDA_SUCCESS) { fprintf(stderr, "[FAIL] cuInit: %d\n", cuErr); return 1; }

    CUdevice cuDev;
    cuErr = cuDeviceGet(&cuDev, 0);
    if (cuErr != CUDA_SUCCESS) { fprintf(stderr, "[FAIL] cuDeviceGet: %d\n", cuErr); return 1; }

    CUcontext cuCtx;
    cuErr = cuDevicePrimaryCtxRetain(&cuCtx, cuDev);
    if (cuErr != CUDA_SUCCESS) { fprintf(stderr, "[FAIL] cuDevicePrimaryCtxRetain: %d\n", cuErr); return 1; }
    cuCtxSetCurrent(cuCtx);

    // -----------------------------------------------------------------------
    // Synthetic frame generation
    // -----------------------------------------------------------------------
    std::vector<uint8_t> frame0(W * H), frame1(W * H);

    // Frame 0: checkerboard with 32-pixel squares
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            frame0[y * W + x] = ((x / 32 + y / 32) % 2) ? 200u : 0u;

    // Frame 1: frame0 shifted +SHIFT_X right, +SHIFT_Y down
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int srcX = x - SHIFT_X;
            int srcY = y - SHIFT_Y;
            frame1[y * W + x] = (srcX >= 0 && srcX < W && srcY >= 0 && srcY < H)
                                 ? frame0[srcY * W + srcX]
                                 : 0u;
        }
    }

    // -----------------------------------------------------------------------
    // Run OFA
    // -----------------------------------------------------------------------
    try {
        OFAPipeline ofa(cuCtx, W, H,
                        NV_OF_OUTPUT_VECTOR_GRID_SIZE_4,
                        NV_OF_PERF_LEVEL_MEDIUM);

        ofa.loadFrame(0, frame1.data());  // slot 0 = inputFrame    (current)
        ofa.loadFrame(1, frame0.data());  // slot 1 = referenceFrame (previous)
        ofa.execute();

        // Synchronize before reading results
        CUresult syncErr = cuCtxSynchronize();
        if (syncErr != CUDA_SUCCESS) {
            fprintf(stderr, "[FAIL] cuCtxSynchronize: %d\n", syncErr);
            return 1;
        }

        const NV_OF_FLOW_VECTOR* vectors = ofa.outputData();

        // -----------------------------------------------------------------------
        // Validation: central 50x50 region, ≥95% within tolerance
        // -----------------------------------------------------------------------
        int pass_count = 0, total = 0;
        int first_fail_x = -1, first_fail_y = -1;
        int16_t first_fail_fx = 0, first_fail_fy = 0;

        for (int y = ROI_Y0; y < ROI_Y0 + ROI_SIZE; ++y) {
            for (int x = ROI_X0; x < ROI_X0 + ROI_SIZE; ++x) {
                const NV_OF_FLOW_VECTOR& v = vectors[y * OUT_W + x];
                bool ok = (std::abs(v.flowx - EXPECTED_FX) <= TOLERANCE) &&
                          (std::abs(v.flowy - EXPECTED_FY) <= TOLERANCE);
                if (ok) {
                    ++pass_count;
                } else if (first_fail_x < 0) {
                    first_fail_x = x; first_fail_y = y;
                    first_fail_fx = v.flowx; first_fail_fy = v.flowy;
                }
                ++total;
            }
        }

        float pass_pct = 100.0f * pass_count / total;
        bool passed = (pass_pct >= 95.0f);

        if (passed) {
            printf("[PASS] OFA verified: %d/%d central vectors within tolerance (%.1f%%)\n",
                   pass_count, total, pass_pct);
        } else {
            printf("[FAIL] Only %d/%d vectors within tolerance (%.1f%%)\n",
                   pass_count, total, pass_pct);
            printf("       First fail at output (%d,%d): flowx=%d (exp %d), flowy=%d (exp %d)\n",
                   first_fail_x, first_fail_y,
                   first_fail_fx, EXPECTED_FX,
                   first_fail_fy, EXPECTED_FY);
        }

        // -----------------------------------------------------------------------
        // PNG output: red = X flow, green = Y flow (±16px range → 0-255)
        // -----------------------------------------------------------------------
        std::vector<uint8_t> img(OUT_W * OUT_H * 3);
        for (int i = 0; i < OUT_W * OUT_H; ++i) {
            float fx = vectors[i].flowx / 32.0f;  // S10.5 → pixels
            float fy = vectors[i].flowy / 32.0f;
            auto encode = [](float v) -> uint8_t {
                float norm = (v + 16.0f) / 32.0f;  // map [-16,+16] px → [0,1]
                return static_cast<uint8_t>(std::clamp(norm * 255.0f, 0.0f, 255.0f));
            };
            img[i * 3 + 0] = encode(fx);
            img[i * 3 + 1] = encode(fy);
            img[i * 3 + 2] = 0;
        }
        stbi_write_png("ofa-test-output.png", OUT_W, OUT_H, 3, img.data(), OUT_W * 3);
        printf("ofa-test-output.png written (%dx%d)\n", OUT_W, OUT_H);

        cuDevicePrimaryCtxRelease(cuDev);
        return passed ? 0 : 1;

    } catch (const std::exception& e) {
        fprintf(stderr, "[FAIL] Exception: %s\n", e.what());
        cuDevicePrimaryCtxRelease(cuDev);
        return 1;
    }
}
```

- [ ] **Step 3: Verify test compiles (will fail to link — OFAPipeline not yet implemented)**

```powershell
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /t:ofa-test /p:Configuration=Debug /p:Platform=x64 /m 2>&1 | Select-String 'error' | Select-Object -First 30"
```

Expected: **linker** errors about unresolved `OFAPipeline` symbols. Compiler errors are not expected — if the file fails to compile (not link), fix before continuing.

- [ ] **Step 4: Commit**

```bash
git add ofa-test/main.cpp ofa-test/stb_image_write.h
git commit -m "feat: add ofa-test harness with synthetic frame validation and PNG output"
```

---

## Task 4: Implement ofa_pipeline.cpp stub (makes it link; validation fails)

**Files:**
- Create: `openxr-api-layer/ofa_pipeline.cpp`

- [ ] **Step 1: Create the stub**

```cpp
// openxr-api-layer/ofa_pipeline.cpp
// Note: uses NvOF and CUDA headers directly — must NOT use the project PCH.

#include "ofa_pipeline.h"

#include <cstring>

OFAPipeline::OFAPipeline(CUcontext /*ctx*/,
                         uint32_t  width,
                         uint32_t  height,
                         NV_OF_OUTPUT_VECTOR_GRID_SIZE gridSize,
                         NV_OF_PERF_LEVEL /*perfLevel*/)
    : m_width(width)
    , m_height(height)
    , m_outW(width  / static_cast<uint32_t>(gridSize))
    , m_outH(height / static_cast<uint32_t>(gridSize))
    , m_hostOutput(m_outW * m_outH, NV_OF_FLOW_VECTOR{0, 0})
{
    // TODO: full init in Task 5
}

OFAPipeline::~OFAPipeline() { destroy(); }

void OFAPipeline::destroy() noexcept {}

void OFAPipeline::loadFrame(int /*slot*/, const void* /*hostGray8*/) {}

void OFAPipeline::execute(CUstream /*stream*/) {}
```

- [ ] **Step 2: Build both projects — both must produce executables**

```powershell
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m 2>&1 | Select-String 'error' | Select-Object -First 30"
```

Expected: **zero errors**. `bin\x64\Debug\ofa-test.exe` exists. Warnings are fine.

- [ ] **Step 3: Run the stub — confirm it reaches the [FAIL] path (not a crash)**

```bash
bin/x64/Debug/ofa-test.exe
```

Expected output (stub returns zero vectors):
```
[FAIL] Only 0/2500 vectors within tolerance (0.0%)
       First fail at output (7,7): flowx=0 (exp 256), flowy=0 (exp 128)
ofa-test-output.png written (64x64)
```

A crash here means a setup problem (CUDA context, etc.) — fix before continuing.

- [ ] **Step 4: Commit**

```bash
git add openxr-api-layer/ofa_pipeline.cpp
git commit -m "feat: add OFAPipeline stub (links; validation not yet implemented)"
```

---

## Task 5: Implement OFAPipeline fully

**Files:**
- Modify: `openxr-api-layer/ofa_pipeline.cpp`

- [ ] **Step 1: Replace stub with full implementation**

```cpp
// openxr-api-layer/ofa_pipeline.cpp
// Note: uses NvOF and CUDA headers directly — must NOT use the project PCH.

#include "ofa_pipeline.h"

#include <cstring>

// -------------------------------------------------------------------------
// Constructor
// -------------------------------------------------------------------------
OFAPipeline::OFAPipeline(CUcontext ctx,
                         uint32_t  width,
                         uint32_t  height,
                         NV_OF_OUTPUT_VECTOR_GRID_SIZE gridSize,
                         NV_OF_PERF_LEVEL              perfLevel)
    : m_width(width)
    , m_height(height)
    , m_outW(width  / static_cast<uint32_t>(gridSize))
    , m_outH(height / static_cast<uint32_t>(gridSize))
    , m_hostOutput(m_outW * m_outH)
{
    // 1. Load NvOf function pointer table
    m_api.size = sizeof(m_api);
    CHECK_NVOF_INIT(NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &m_api));

    // 2. Create OFA instance bound to CUDA context
    CHECK_NVOF(m_api.nvCreateOpticalFlowCuda(ctx, &m_hOf));

    // 3. Configure and initialise OFA
    NV_OF_INIT_PARAMS params{};
    params.width             = width;
    params.height            = height;
    params.outGridSize       = gridSize;
    params.hintGridSize      = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
    params.mode              = NV_OF_MODE_OPTICALFLOW;
    params.perfLevel         = perfLevel;
    params.enableExternalHints = NV_OF_FALSE;
    params.enableOutputCost  = NV_OF_FALSE;
    params.enableRoi         = NV_OF_FALSE;
    params.predDirection     = NV_OF_PRED_DIRECTION_FORWARD;
    params.enableGlobalFlow  = NV_OF_FALSE;
    params.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    CHECK_NVOF(m_api.nvOFInit(m_hOf, &params));

    // 4. Create two input buffers (slot 0 = current, slot 1 = reference)
    NV_OF_BUFFER_DESCRIPTOR inputDesc{};
    inputDesc.width       = width;
    inputDesc.height      = height;
    inputDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
    inputDesc.bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    for (int i = 0; i < 2; ++i)
        CHECK_NVOF(m_api.nvOFCreateGPUBufferCuda(m_hOf, &inputDesc,
                   NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_inputBufs[i]));

    // 5. Create output buffer (NV_OF_FLOW_VECTOR = SHORT2, one entry per grid cell)
    NV_OF_BUFFER_DESCRIPTOR outputDesc{};
    outputDesc.width       = m_outW;
    outputDesc.height      = m_outH;
    outputDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    outputDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
    CHECK_NVOF(m_api.nvOFCreateGPUBufferCuda(m_hOf, &outputDesc,
               NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &m_outputBuf));
}

// -------------------------------------------------------------------------
// Destructor
// -------------------------------------------------------------------------
OFAPipeline::~OFAPipeline() { destroy(); }

void OFAPipeline::destroy() noexcept {
    // GPU buffer handles before OFA instance (reverse init order)
    if (m_outputBuf)    { m_api.nvOFDestroyGPUBufferCuda(m_outputBuf);    m_outputBuf = nullptr; }
    if (m_inputBufs[1]) { m_api.nvOFDestroyGPUBufferCuda(m_inputBufs[1]); m_inputBufs[1] = nullptr; }
    if (m_inputBufs[0]) { m_api.nvOFDestroyGPUBufferCuda(m_inputBufs[0]); m_inputBufs[0] = nullptr; }
    if (m_hOf)          { m_api.nvOFDestroy(m_hOf);                        m_hOf = nullptr; }
}

// -------------------------------------------------------------------------
// loadFrame
// -------------------------------------------------------------------------
void OFAPipeline::loadFrame(int slot, const void* hostGray8) {
    // Get stride info so we respect any pitch alignment OFA chose for the buffer.
    NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
    CHECK_NVOF(m_api.nvOFGPUBufferGetStrideInfo(m_inputBufs[slot], &si));

    CUdeviceptr devPtr = m_api.nvOFGPUBufferGetCUdeviceptr(m_inputBufs[slot]);

    CUDA_MEMCPY2D cp{};
    cp.srcMemoryType  = CU_MEMORYTYPE_HOST;
    cp.srcHost        = hostGray8;
    cp.srcPitch       = m_width;                          // tightly packed host data
    cp.dstMemoryType  = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice      = devPtr;
    cp.dstPitch       = si.strideInfo[0].strideXInBytes;  // OFA-allocated pitch
    cp.WidthInBytes   = m_width;                          // 1 byte per pixel (GRAYSCALE8)
    cp.Height         = m_height;
    CHECK_CU(cuMemcpy2D(&cp));
}

// -------------------------------------------------------------------------
// execute
// -------------------------------------------------------------------------
void OFAPipeline::execute(CUstream /*stream*/) {
    // `stream` is reserved for Item 4 pre-warp integration.
    // OFA uses its own internal stream; the caller must cuCtxSynchronize() after.

    NV_OF_EXECUTE_INPUT_PARAMS in{};
    in.inputFrame          = m_inputBufs[0];  // current frame
    in.referenceFrame      = m_inputBufs[1];  // previous frame
    in.disableTemporalHints = NV_OF_TRUE;      // no inter-frame predictor for test

    NV_OF_EXECUTE_OUTPUT_PARAMS out{};
    out.outputBuffer = m_outputBuf;

    CHECK_NVOF(m_api.nvOFExecute(m_hOf, &in, &out));

    // Copy motion vector output to host staging buffer.
    // Caller is responsible for cuCtxSynchronize() before reading outputData().
    NV_OF_CUDA_BUFFER_STRIDE_INFO si{};
    CHECK_NVOF(m_api.nvOFGPUBufferGetStrideInfo(m_outputBuf, &si));

    CUdeviceptr outDev = m_api.nvOFGPUBufferGetCUdeviceptr(m_outputBuf);
    const uint32_t rowBytes = m_outW * static_cast<uint32_t>(sizeof(NV_OF_FLOW_VECTOR));

    CUDA_MEMCPY2D cp{};
    cp.srcMemoryType  = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice      = outDev;
    cp.srcPitch       = si.strideInfo[0].strideXInBytes;
    cp.dstMemoryType  = CU_MEMORYTYPE_HOST;
    cp.dstHost        = m_hostOutput.data();
    cp.dstPitch       = rowBytes;
    cp.WidthInBytes   = rowBytes;
    cp.Height         = m_outH;
    CHECK_CU(cuMemcpy2D(&cp));
}
```

- [ ] **Step 2: Build the full solution**

```powershell
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m 2>&1 | Select-String 'error' | Select-Object -First 30"
```

Expected: zero errors. If there are compile errors related to missing enum values (e.g. `NV_OF_PRED_DIRECTION_FORWARD` or `NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED` not found), open `C:\Project\Optical_Flow_SDK_5.0.7\NvOFInterface\nvOpticalFlowCommon.h` and find the correct enum value names to substitute.

- [ ] **Step 3: Run ofa-test.exe**

```bash
bin/x64/Debug/ofa-test.exe
```

Expected:
```
[PASS] OFA verified: XXXX/2500 central vectors within tolerance (≥95.0%)
ofa-test-output.png written (64x64)
```

If `[FAIL]`: open `ofa-test-output.png` and inspect. A uniform greenish color (both channels near midpoint ≈128) means zero vectors — OFA likely received blank frames; verify `loadFrame` pitch math. A random-noise image means frames were uploaded correctly but OFA didn't find a clear match — check `inputFrame`/`referenceFrame` slot assignment (slot 0 must be current/shifted, slot 1 must be reference/original).

- [ ] **Step 4: Verify openxr-api-layer still builds cleanly**

```powershell
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' openxr-api-layer\openxr-api-layer.vcxproj /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m 2>&1 | Select-String 'error' | Select-Object -First 10"
```

Expected: zero errors.

- [ ] **Step 5: Commit**

```bash
git add openxr-api-layer/ofa_pipeline.cpp
git commit -m "feat: implement OFAPipeline — NvOF init, loadFrame, execute (Roadmap Item 3)"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run Release build end-to-end**

```powershell
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Release /p:Platform=x64 /m 2>&1 | Select-String 'error' | Select-Object -First 20"
```

Expected: zero errors.

```bash
bin/x64/Release/ofa-test.exe
```

Expected: `[PASS]` with ≥95%.

- [ ] **Step 2: Visually inspect ofa-test-output.png**

Open `ofa-test-output.png` (64×64). Expected appearance: a **uniform reddish-orange block** — red channel ≈ 191 (encoding +8px), green channel ≈ 160 (encoding +4px), with a small darker border on one or two edges (border artifact region). If the interior is that uniform color, OFA is working correctly.

Encoding reference:
- +8px in X → `encode(+8) = (8+16)/32*255 ≈ 191` → red
- +4px in Y → `encode(+4) = (4+16)/32*255 ≈ 159` → green

- [ ] **Step 3: Final commit and tag**

```bash
git add -A
git commit -m "feat: complete Roadmap Item 3 — OFA integration validated on RTX 5070 Ti"
```

---

## Verification Checklist (from spec)

- [ ] `ofa-test.exe` prints `[PASS]` on RTX 5070 Ti
- [ ] `ofa-test-output.png` shows uniform reddish-orange in the central region
- [ ] `openxr-api-layer` project builds clean with zero new errors
- [ ] NvOf headers do not leak into `pch.h`
