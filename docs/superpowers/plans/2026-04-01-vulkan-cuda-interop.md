# Vulkan/CUDA Interop Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `SharedImage` and `SharedSemaphore` RAII types that share GPU resources between Vulkan and CUDA, validated by a standalone console test that writes a known pattern via CUDA and reads it back via Vulkan.

**Architecture:** A new `vulkan_cuda_interop` module in `openxr-api-layer/` declares two move-only RAII classes. A separate `interop-test` console project references the same `.cpp` directly and exercises the full round-trip: CUDA kernel write → semaphore signal → Vulkan readback → pixel verification.

**Tech Stack:** C++17, Vulkan 1.4 (driver API via `vulkan-1.lib`), CUDA 13.2 driver API (`cuda.lib`), CUDA runtime for `.cu` kernel (`cudart.lib`), MSBuild/CUDA 13.2 build customization.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `openxr-api-layer/vulkan_cuda_interop.h` | Create | `SharedImage` and `SharedSemaphore` declarations |
| `openxr-api-layer/vulkan_cuda_interop.cpp` | Create | Full RAII implementations |
| `openxr-api-layer/openxr-api-layer.vcxproj` | Modify | Add CUDA include/lib; add `vulkan_cuda_interop.cpp` with PCH override |
| `interop-test/interop-test.vcxproj` | Create | Console app project with CUDA build integration |
| `interop-test/fill_pattern.cu` | Create | CUDA kernel: writes `(x + y*width) % 256` into R channel of each pixel |
| `interop-test/main.cpp` | Create | Headless Vulkan setup, test orchestration, pixel verification |
| `SMOOTHING-OPENXR-LAYER.sln` | Modify | Add `interop-test` project |

---

## Task 1: Add CUDA driver API to `openxr-api-layer.vcxproj`

**Files:**
- Modify: `openxr-api-layer/openxr-api-layer.vcxproj:103-144` (Debug|x64 ItemDefinitionGroup)
- Modify: `openxr-api-layer/openxr-api-layer.vcxproj:188-234` (Release|x64 ItemDefinitionGroup)
- Modify: `openxr-api-layer/openxr-api-layer.vcxproj:293-310` (ClCompile ItemGroup)

- [ ] **Step 1: Add CUDA include path to Debug|x64 `AdditionalIncludeDirectories`**

In `openxr-api-layer/openxr-api-layer.vcxproj`, find:
```xml
      <AdditionalIncludeDirectories>C:\VulkanSDK\1.4.341.1\Include;$(ProjectDir);$(ProjectDir)\framework;$(SolutionDir)\external\OpenXR-SDK\include;$(SolutionDir)\external\OpenXR-SDK\src\common;$(SolutionDir)\external\OpenXR-MixedReality\Shared\XrUtility</AdditionalIncludeDirectories>
      <LanguageStandard>stdcpp17</LanguageStandard>
    </ClCompile>
    <Link>
      <SubSystem>Windows</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableUAC>false</EnableUAC>
      <AdditionalDependencies>C:\VulkanSDK\1.4.341.1\Lib\vulkan-1.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>
```
(this is in the `Condition="'$(Configuration)|$(Platform)'=='Debug|x64'"` ItemDefinitionGroup)

Replace with:
```xml
      <AdditionalIncludeDirectories>$(CudaToolkitDir)include;C:\VulkanSDK\1.4.341.1\Include;$(ProjectDir);$(ProjectDir)\framework;$(SolutionDir)\external\OpenXR-SDK\include;$(SolutionDir)\external\OpenXR-SDK\src\common;$(SolutionDir)\external\OpenXR-MixedReality\Shared\XrUtility</AdditionalIncludeDirectories>
      <LanguageStandard>stdcpp17</LanguageStandard>
    </ClCompile>
    <Link>
      <SubSystem>Windows</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableUAC>false</EnableUAC>
      <AdditionalDependencies>cuda.lib;C:\VulkanSDK\1.4.341.1\Lib\vulkan-1.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>
```

- [ ] **Step 2: Add CUDA paths to Release|x64 ItemDefinitionGroup**

In `openxr-api-layer/openxr-api-layer.vcxproj`, find (inside `Condition="'$(Configuration)|$(Platform)'=='Release|x64'"` ItemDefinitionGroup):
```xml
      <AdditionalIncludeDirectories>C:\VulkanSDK\1.4.341.1\Include;$(ProjectDir);$(ProjectDir)\framework;$(SolutionDir)\external\OpenXR-SDK\include;$(SolutionDir)\external\OpenXR-SDK\src\common;$(SolutionDir)\external\OpenXR-MixedReality\Shared\XrUtility</AdditionalIncludeDirectories>
      <LanguageStandard>stdcpp17</LanguageStandard>
    </ClCompile>
    <Link>
      <SubSystem>Windows</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableUAC>false</EnableUAC>
      <AdditionalDependencies>C:\VulkanSDK\1.4.341.1\Lib\vulkan-1.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>
```

Replace with:
```xml
      <AdditionalIncludeDirectories>$(CudaToolkitDir)include;C:\VulkanSDK\1.4.341.1\Include;$(ProjectDir);$(ProjectDir)\framework;$(SolutionDir)\external\OpenXR-SDK\include;$(SolutionDir)\external\OpenXR-SDK\src\common;$(SolutionDir)\external\OpenXR-MixedReality\Shared\XrUtility</AdditionalIncludeDirectories>
      <LanguageStandard>stdcpp17</LanguageStandard>
    </ClCompile>
    <Link>
      <SubSystem>Windows</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableUAC>false</EnableUAC>
      <AdditionalDependencies>cuda.lib;C:\VulkanSDK\1.4.341.1\Lib\vulkan-1.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies)</AdditionalDependencies>
```

- [ ] **Step 3: Add `vulkan_cuda_interop.cpp` to ClCompile with PCH disabled**

In `openxr-api-layer/openxr-api-layer.vcxproj`, find:
```xml
    <ClCompile Include="utils\composition.cpp" />
```

Replace with:
```xml
    <ClCompile Include="utils\composition.cpp" />
    <ClCompile Include="vulkan_cuda_interop.cpp">
      <PrecompiledHeader Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">NotUsing</PrecompiledHeader>
      <PrecompiledHeader Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">NotUsing</PrecompiledHeader>
      <PrecompiledHeader Condition="'$(Configuration)|$(Platform)'=='Release|x64'">NotUsing</PrecompiledHeader>
      <PrecompiledHeader Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">NotUsing</PrecompiledHeader>
    </ClCompile>
```

- [ ] **Step 4: Add `vulkan_cuda_interop.h` to ClInclude**

In `openxr-api-layer/openxr-api-layer.vcxproj`, find:
```xml
    <ClInclude Include="layer.h" />
```

Replace with:
```xml
    <ClInclude Include="layer.h" />
    <ClInclude Include="vulkan_cuda_interop.h" />
```

---

## Task 2: Create `vulkan_cuda_interop.h`

**Files:**
- Create: `openxr-api-layer/vulkan_cuda_interop.h`

- [ ] **Step 1: Write the header**

Create `openxr-api-layer/vulkan_cuda_interop.h` with this content:

```cpp
#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

namespace interop {

// RAII wrapper for a Vulkan image whose memory is exported to CUDA.
//
// Construction allocates a VkImage with VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
// exports the Win32 handle, and imports it into CUDA as a CUarray.
//
// CUDA must be initialized (cuInit called, context current) before constructing.
// VK_KHR_external_memory_win32 must be enabled on the VkDevice.
class SharedImage {
  public:
    SharedImage(VkDevice device, VkPhysicalDevice physDevice,
                uint32_t width, uint32_t height, VkFormat format);
    ~SharedImage();

    SharedImage(const SharedImage&) = delete;
    SharedImage& operator=(const SharedImage&) = delete;
    SharedImage(SharedImage&&) noexcept;
    SharedImage& operator=(SharedImage&&) noexcept;

    VkImage  vkImage()  const { return m_image; }
    CUarray  cuArray()  const { return m_cuArray; }
    uint32_t width()    const { return m_width; }
    uint32_t height()   const { return m_height; }

  private:
    VkDevice          m_device{VK_NULL_HANDLE};
    VkImage           m_image{VK_NULL_HANDLE};
    VkDeviceMemory    m_memory{VK_NULL_HANDLE};
    CUexternalMemory  m_extMem{nullptr};
    CUmipmappedArray  m_mipmappedArray{nullptr};
    CUarray           m_cuArray{nullptr};
    uint32_t          m_width{0};
    uint32_t          m_height{0};

    void destroy() noexcept;
};

// RAII wrapper for a Vulkan semaphore exported to CUDA.
//
// Allows CUDA to signal/wait on the same semaphore that Vulkan
// uses in VkSubmitInfo::pWaitSemaphores / pSignalSemaphores.
//
// VK_KHR_external_semaphore_win32 must be enabled on the VkDevice.
class SharedSemaphore {
  public:
    explicit SharedSemaphore(VkDevice device);
    ~SharedSemaphore();

    SharedSemaphore(const SharedSemaphore&) = delete;
    SharedSemaphore& operator=(const SharedSemaphore&) = delete;
    SharedSemaphore(SharedSemaphore&&) noexcept;
    SharedSemaphore& operator=(SharedSemaphore&&) noexcept;

    // Vulkan handle — use in VkSubmitInfo::pWaitSemaphores or pSignalSemaphores.
    VkSemaphore vkSemaphore() const { return m_semaphore; }

    // Enqueues a signal on `stream`. Vulkan can then wait on vkSemaphore().
    void signal(CUstream stream);

    // Enqueues a wait on `stream`. CUDA execution stalls until Vulkan signals.
    void wait(CUstream stream);

  private:
    VkDevice             m_device{VK_NULL_HANDLE};
    VkSemaphore          m_semaphore{VK_NULL_HANDLE};
    CUexternalSemaphore  m_extSem{nullptr};

    void destroy() noexcept;
};

} // namespace interop
```

---

## Task 3: Implement `SharedImage` in `vulkan_cuda_interop.cpp`

**Files:**
- Create: `openxr-api-layer/vulkan_cuda_interop.cpp`

- [ ] **Step 1: Write the file up through `SharedImage` (constructor + destructor + move)**

Create `openxr-api-layer/vulkan_cuda_interop.cpp`:

```cpp
#define VK_USE_PLATFORM_WIN32_KHR
#include "vulkan_cuda_interop.h"

#include <vulkan/vulkan_win32.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define CHECK_VK(call)                                                          \
    do {                                                                        \
        VkResult _r = (call);                                                   \
        if (_r != VK_SUCCESS)                                                   \
            throw std::runtime_error(std::string("Vulkan error ") +             \
                                     std::to_string(_r) + " in " #call);       \
    } while (0)

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        CUresult _r = (call);                                                   \
        if (_r != CUDA_SUCCESS) {                                               \
            const char* str = nullptr;                                          \
            cuGetErrorString(_r, &str);                                         \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     (str ? str : "unknown") + " in " #call);  \
        }                                                                       \
    } while (0)

namespace interop {

// ---------------------------------------------------------------------------
// SharedImage
// ---------------------------------------------------------------------------

SharedImage::SharedImage(VkDevice device, VkPhysicalDevice physDevice,
                         uint32_t width, uint32_t height, VkFormat format)
    : m_device(device), m_width(width), m_height(height) {

    // Load extension function — not linked directly, must be fetched at runtime.
    auto vkGetMemoryWin32HandleKHR =
        reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    if (!vkGetMemoryWin32HandleKHR)
        throw std::runtime_error(
            "vkGetMemoryWin32HandleKHR not available — "
            "VK_KHR_external_memory_win32 must be enabled at VkDevice creation");

    // 1. Create VkImage with external memory export flag.
    VkExternalMemoryImageCreateInfo extImageInfo{};
    extImageInfo.sType      = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    extImageInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext         = &extImageInfo;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = format;
    imageInfo.extent        = {width, height, 1};
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_STORAGE_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    CHECK_VK(vkCreateImage(device, &imageInfo, nullptr, &m_image));

    // 2. Find a device-local memory type that satisfies the image requirements.
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, m_image, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);

    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    if (memTypeIndex == UINT32_MAX)
        throw std::runtime_error("No device-local memory type satisfies image requirements");

    // 3. Allocate memory with Win32 export flag.
    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext           = &exportInfo;
    allocInfo.allocationSize  = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    CHECK_VK(vkAllocateMemory(device, &allocInfo, nullptr, &m_memory));
    CHECK_VK(vkBindImageMemory(device, m_image, m_memory, 0));

    // 4. Export the Win32 handle.
    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory     = m_memory;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE win32Handle    = nullptr;
    CHECK_VK(vkGetMemoryWin32HandleKHR(device, &handleInfo, &win32Handle));

    // 5. Import into CUDA.
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC extMemDesc{};
    extMemDesc.type                = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    extMemDesc.handle.win32.handle = win32Handle;
    extMemDesc.size                = memReqs.size;
    CUresult cuResult = cuImportExternalMemory(&m_extMem, &extMemDesc);
    CloseHandle(win32Handle);  // Always close — CUDA takes ownership of the import.
    CHECK_CUDA(cuResult);

    // 6. Map to a CUarray (mip level 0).
    //    CUDA_ARRAY3D_SURFACE_LDST is required for surf2Dwrite in kernels.
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC arrayDesc{};
    arrayDesc.offset               = 0;
    arrayDesc.arrayDesc.Width      = width;
    arrayDesc.arrayDesc.Height     = height;
    arrayDesc.arrayDesc.Depth      = 0;
    arrayDesc.arrayDesc.Format     = CU_AD_FORMAT_UNSIGNED_INT8;
    arrayDesc.arrayDesc.NumChannels = 4;  // RGBA
    arrayDesc.arrayDesc.Flags      = CUDA_ARRAY3D_SURFACE_LDST;
    arrayDesc.numLevels            = 1;
    CHECK_CUDA(cuExternalMemoryGetMappedMipmappedArray(&m_mipmappedArray, m_extMem, &arrayDesc));
    CHECK_CUDA(cuMipmappedArrayGetLevel(&m_cuArray, m_mipmappedArray, 0));
}

SharedImage::~SharedImage() {
    destroy();
}

void SharedImage::destroy() noexcept {
    if (m_cuArray)         { m_cuArray = nullptr; }  // owned by m_mipmappedArray
    if (m_mipmappedArray)  { cuMipmappedArrayDestroy(m_mipmappedArray); m_mipmappedArray = nullptr; }
    if (m_extMem)          { cuDestroyExternalMemory(m_extMem); m_extMem = nullptr; }
    if (m_memory)          { vkFreeMemory(m_device, m_memory, nullptr); m_memory = VK_NULL_HANDLE; }
    if (m_image)           { vkDestroyImage(m_device, m_image, nullptr); m_image = VK_NULL_HANDLE; }
}

SharedImage::SharedImage(SharedImage&& other) noexcept
    : m_device(other.m_device), m_image(other.m_image), m_memory(other.m_memory),
      m_extMem(other.m_extMem), m_mipmappedArray(other.m_mipmappedArray),
      m_cuArray(other.m_cuArray), m_width(other.m_width), m_height(other.m_height) {
    other.m_device = VK_NULL_HANDLE;
    other.m_image  = VK_NULL_HANDLE;
    other.m_memory = VK_NULL_HANDLE;
    other.m_extMem = nullptr;
    other.m_mipmappedArray = nullptr;
    other.m_cuArray = nullptr;
}

SharedImage& SharedImage::operator=(SharedImage&& other) noexcept {
    if (this != &other) {
        destroy();
        m_device        = other.m_device;
        m_image         = other.m_image;
        m_memory        = other.m_memory;
        m_extMem        = other.m_extMem;
        m_mipmappedArray = other.m_mipmappedArray;
        m_cuArray       = other.m_cuArray;
        m_width         = other.m_width;
        m_height        = other.m_height;
        other.m_device        = VK_NULL_HANDLE;
        other.m_image         = VK_NULL_HANDLE;
        other.m_memory        = VK_NULL_HANDLE;
        other.m_extMem        = nullptr;
        other.m_mipmappedArray = nullptr;
        other.m_cuArray       = nullptr;
    }
    return *this;
}
```

---

## Task 4: Implement `SharedSemaphore` in `vulkan_cuda_interop.cpp`

**Files:**
- Modify: `openxr-api-layer/vulkan_cuda_interop.cpp` (append to end of file)

- [ ] **Step 1: Append `SharedSemaphore` implementation**

Append to the end of `openxr-api-layer/vulkan_cuda_interop.cpp`:

```cpp
// ---------------------------------------------------------------------------
// SharedSemaphore
// ---------------------------------------------------------------------------

SharedSemaphore::SharedSemaphore(VkDevice device) : m_device(device) {
    auto vkGetSemaphoreWin32HandleKHR =
        reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
    if (!vkGetSemaphoreWin32HandleKHR)
        throw std::runtime_error(
            "vkGetSemaphoreWin32HandleKHR not available — "
            "VK_KHR_external_semaphore_win32 must be enabled at VkDevice creation");

    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semInfo.pNext = &exportInfo;
    CHECK_VK(vkCreateSemaphore(device, &semInfo, nullptr, &m_semaphore));

    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore  = m_semaphore;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE win32Handle = nullptr;
    CHECK_VK(vkGetSemaphoreWin32HandleKHR(device, &handleInfo, &win32Handle));

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC extSemDesc{};
    extSemDesc.type                = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
    extSemDesc.handle.win32.handle = win32Handle;
    CUresult cuResult = cuImportExternalSemaphore(&m_extSem, &extSemDesc);
    CloseHandle(win32Handle);
    CHECK_CUDA(cuResult);
}

SharedSemaphore::~SharedSemaphore() {
    destroy();
}

void SharedSemaphore::destroy() noexcept {
    if (m_extSem)     { cuDestroyExternalSemaphore(m_extSem); m_extSem = nullptr; }
    if (m_semaphore)  { vkDestroySemaphore(m_device, m_semaphore, nullptr); m_semaphore = VK_NULL_HANDLE; }
}

SharedSemaphore::SharedSemaphore(SharedSemaphore&& other) noexcept
    : m_device(other.m_device), m_semaphore(other.m_semaphore), m_extSem(other.m_extSem) {
    other.m_device    = VK_NULL_HANDLE;
    other.m_semaphore = VK_NULL_HANDLE;
    other.m_extSem    = nullptr;
}

SharedSemaphore& SharedSemaphore::operator=(SharedSemaphore&& other) noexcept {
    if (this != &other) {
        destroy();
        m_device    = other.m_device;
        m_semaphore = other.m_semaphore;
        m_extSem    = other.m_extSem;
        other.m_device    = VK_NULL_HANDLE;
        other.m_semaphore = VK_NULL_HANDLE;
        other.m_extSem    = nullptr;
    }
    return *this;
}

void SharedSemaphore::signal(CUstream stream) {
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
    CHECK_CUDA(cuSignalExternalSemaphoresAsync(&m_extSem, &params, 1, stream));
}

void SharedSemaphore::wait(CUstream stream) {
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
    CHECK_CUDA(cuWaitExternalSemaphoresAsync(&m_extSem, &params, 1, stream));
}

} // namespace interop
```

- [ ] **Step 2: Commit**

```bash
git add openxr-api-layer/vulkan_cuda_interop.h \
        openxr-api-layer/vulkan_cuda_interop.cpp \
        openxr-api-layer/openxr-api-layer.vcxproj
git commit -m "feat: add vulkan_cuda_interop module (SharedImage, SharedSemaphore)"
```

---

## Task 5: Build the layer — verify `vulkan_cuda_interop` compiles

**Files:** (build only, no edits)

- [ ] **Step 1: Run debug build**

```
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m"
```

Expected: `Build succeeded.` with 0 errors. One pre-existing LNK4099 warning about `fmt.pdb` is acceptable. Any other error means a fix is needed before continuing.

---

## Task 6: Create the `interop-test` project and add to solution

**Files:**
- Create: `interop-test/interop-test.vcxproj`
- Modify: `SMOOTHING-OPENXR-LAYER.sln`

- [ ] **Step 1: Create `interop-test/interop-test.vcxproj`**

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
    <VCProjectVersion>16.0</VCProjectVersion>
    <ProjectGuid>{1A2B3C4D-5E6F-7A8B-9C0D-E1F2A3B4C5D6}</ProjectGuid>
    <RootNamespace>interoptest</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v145</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v145</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ImportGroup Label="ExtensionSettings">
    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.2.props" />
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props"
            Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')"
            Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props"
            Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')"
            Label="LocalAppDataPlatform" />
  </ImportGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <OutDir>$(SolutionDir)bin\$(Platform)\$(Configuration)\</OutDir>
    <IntDir>$(SolutionDir)obj\$(Platform)\$(Configuration)\interop-test\</IntDir>
    <TargetName>interop-test</TargetName>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <OutDir>$(SolutionDir)bin\$(Platform)\$(Configuration)\</OutDir>
    <IntDir>$(SolutionDir)obj\$(Platform)\$(Configuration)\interop-test\</IntDir>
    <TargetName>interop-test</TargetName>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>VK_USE_PLATFORM_WIN32_KHR;_DEBUG;_CONSOLE;_CRT_SECURE_NO_WARNINGS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <AdditionalIncludeDirectories>$(CudaToolkitDir)include;C:\VulkanSDK\1.4.341.1\Include;$(SolutionDir)openxr-api-layer;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
      <LanguageStandard>stdcpp17</LanguageStandard>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <AdditionalLibraryDirectories>$(CudaToolkitDir)lib\x64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
      <AdditionalDependencies>cuda.lib;cudart.lib;C:\VulkanSDK\1.4.341.1\Lib\vulkan-1.lib;kernel32.lib;user32.lib;%(AdditionalDependencies)</AdditionalDependencies>
    </Link>
    <CudaCompile>
      <TargetMachinePlatform>64</TargetMachinePlatform>
    </CudaCompile>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>VK_USE_PLATFORM_WIN32_KHR;NDEBUG;_CONSOLE;_CRT_SECURE_NO_WARNINGS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <AdditionalIncludeDirectories>$(CudaToolkitDir)include;C:\VulkanSDK\1.4.341.1\Include;$(SolutionDir)openxr-api-layer;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
      <LanguageStandard>stdcpp17</LanguageStandard>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <AdditionalLibraryDirectories>$(CudaToolkitDir)lib\x64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
      <AdditionalDependencies>cuda.lib;cudart.lib;C:\VulkanSDK\1.4.341.1\Lib\vulkan-1.lib;kernel32.lib;user32.lib;%(AdditionalDependencies)</AdditionalDependencies>
    </Link>
    <CudaCompile>
      <TargetMachinePlatform>64</TargetMachinePlatform>
    </CudaCompile>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="main.cpp" />
    <ClCompile Include="..\openxr-api-layer\vulkan_cuda_interop.cpp" />
    <CudaCompile Include="fill_pattern.cu" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionTargets">
    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.2.targets" />
  </ImportGroup>
</Project>
```

- [ ] **Step 2: Add `interop-test` to `SMOOTHING-OPENXR-LAYER.sln`**

In `SMOOTHING-OPENXR-LAYER.sln`, find:
```
Project("{2150E333-8FDC-42A3-9474-1A3956D46DE8}") = "Solution Files"
```

Insert the following **before** that line:
```
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "interop-test", "interop-test\interop-test.vcxproj", "{1A2B3C4D-5E6F-7A8B-9C0D-E1F2A3B4C5D6}"
EndProject
```

Then, in the `GlobalSection(ProjectConfigurationPlatforms)` block, find:
```
		{93D573D0-634F-4BA0-8FE0-FB63D7D00A05}.Release|x64.Build.0 = Release|x64
```

Insert after it:
```
		{1A2B3C4D-5E6F-7A8B-9C0D-E1F2A3B4C5D6}.Debug|x64.ActiveCfg = Debug|x64
		{1A2B3C4D-5E6F-7A8B-9C0D-E1F2A3B4C5D6}.Debug|x64.Build.0 = Debug|x64
		{1A2B3C4D-5E6F-7A8B-9C0D-E1F2A3B4C5D6}.Debug|Win32.ActiveCfg = Debug|x64
		{1A2B3C4D-5E6F-7A8B-9C0D-E1F2A3B4C5D6}.Release|x64.ActiveCfg = Release|x64
		{1A2B3C4D-5E6F-7A8B-9C0D-E1F2A3B4C5D6}.Release|x64.Build.0 = Release|x64
		{1A2B3C4D-5E6F-7A8B-9C0D-E1F2A3B4C5D6}.Release|Win32.ActiveCfg = Release|x64
```

- [ ] **Step 3: Commit**

```bash
git add interop-test/interop-test.vcxproj SMOOTHING-OPENXR-LAYER.sln
git commit -m "feat: add interop-test project to solution"
```

---

## Task 7: Write `fill_pattern.cu`

**Files:**
- Create: `interop-test/fill_pattern.cu`

- [ ] **Step 1: Write the kernel**

Create `interop-test/fill_pattern.cu`:

```cuda
#include <surface_functions.h>
#include <cuda_runtime.h>

// Writes a deterministic pattern to a 2D RGBA surface.
// Each pixel's R channel = (x + y * width) % 256. G, B = 0, A = 255.
// This pattern is verifiable on CPU without floating-point comparison.
extern "C" __global__ void fill_pattern(cudaSurfaceObject_t surf,
                                         int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uchar4 pixel = make_uchar4(
        static_cast<unsigned char>((x + y * width) % 256),
        0,
        0,
        255
    );
    surf2Dwrite(pixel, surf, x * sizeof(uchar4), y);
}
```

- [ ] **Step 2: Commit**

```bash
git add interop-test/fill_pattern.cu
git commit -m "feat: add fill_pattern CUDA kernel for interop test"
```

---

## Task 8: Write `interop-test/main.cpp`

**Files:**
- Create: `interop-test/main.cpp`

- [ ] **Step 1: Write the full test**

Create `interop-test/main.cpp`:

```cpp
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "vulkan_cuda_interop.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declaration of the CUDA kernel (compiled in fill_pattern.cu).
// ---------------------------------------------------------------------------
extern "C" void fill_pattern(cudaSurfaceObject_t surf, int width, int height);

// ---------------------------------------------------------------------------
// Minimal helpers
// ---------------------------------------------------------------------------

static void check_vk(VkResult r, const char* where) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[FAIL] Vulkan error %d at %s\n", r, where);
        exit(1);
    }
}
#define VK(call) check_vk((call), #call)

static void check_cu(CUresult r, const char* where) {
    if (r != CUDA_SUCCESS) {
        const char* str = nullptr;
        cuGetErrorString(r, &str);
        fprintf(stderr, "[FAIL] CUDA error %s at %s\n", str ? str : "?", where);
        exit(1);
    }
}
#define CU(call) check_cu((call), #call)

// ---------------------------------------------------------------------------
// Headless Vulkan setup
// ---------------------------------------------------------------------------

struct VulkanContext {
    VkInstance       instance{VK_NULL_HANDLE};
    VkPhysicalDevice physDevice{VK_NULL_HANDLE};
    VkDevice         device{VK_NULL_HANDLE};
    VkQueue          queue{VK_NULL_HANDLE};
    uint32_t         queueFamily{0};
    VkCommandPool    cmdPool{VK_NULL_HANDLE};
    VkCommandBuffer  cmdBuf{VK_NULL_HANDLE};
};

static VulkanContext createVulkan() {
    VulkanContext ctx;

    // Instance
    const char* instExts[] = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };
    VkApplicationInfo appInfo{};
    appInfo.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo        = &appInfo;
    instInfo.enabledExtensionCount   = 3;
    instInfo.ppEnabledExtensionNames = instExts;
    VK(vkCreateInstance(&instInfo, nullptr, &ctx.instance));

    // Physical device — pick first discrete GPU
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, nullptr);
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, devs.data());

    for (auto pd : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx.physDevice = pd;
            printf("Using GPU: %s\n", props.deviceName);
            break;
        }
    }
    if (!ctx.physDevice) {
        // Fallback: just use whatever is available
        ctx.physDevice = devs[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(ctx.physDevice, &props);
        printf("Using GPU (fallback): %s\n", props.deviceName);
    }

    // Find a queue family that supports compute + transfer
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physDevice, &qfCount, qfs.data());

    ctx.queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (qfs[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) {
            ctx.queueFamily = i;
            break;
        }
    }
    if (ctx.queueFamily == UINT32_MAX)
        throw std::runtime_error("No suitable queue family");

    // Device with external memory + semaphore extensions
    const char* devExts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
    };
    float qPriority = 1.0f;
    VkDeviceQueueCreateInfo qInfo{};
    qInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = ctx.queueFamily;
    qInfo.queueCount       = 1;
    qInfo.pQueuePriorities = &qPriority;

    VkDeviceCreateInfo devInfo{};
    devInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount    = 1;
    devInfo.pQueueCreateInfos       = &qInfo;
    devInfo.enabledExtensionCount   = 4;
    devInfo.ppEnabledExtensionNames = devExts;
    VK(vkCreateDevice(ctx.physDevice, &devInfo, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);

    // Command pool + buffer
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = ctx.queueFamily;
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.cmdPool));

    VkCommandBufferAllocateInfo cbInfo{};
    cbInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool        = ctx.cmdPool;
    cbInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    VK(vkAllocateCommandBuffers(ctx.device, &cbInfo, &ctx.cmdBuf));

    return ctx;
}

static void destroyVulkan(VulkanContext& ctx) {
    if (ctx.cmdPool)   vkDestroyCommandPool(ctx.device, ctx.cmdPool, nullptr);
    if (ctx.device)    vkDestroyDevice(ctx.device, nullptr);
    if (ctx.instance)  vkDestroyInstance(ctx.instance, nullptr);
}

// ---------------------------------------------------------------------------
// Find a host-visible memory type for the readback buffer
// ---------------------------------------------------------------------------
static uint32_t findHostVisibleMemType(VkPhysicalDevice pd, uint32_t typeBits) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            return i;
        }
    }
    throw std::runtime_error("No host-visible memory type");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    const int W = 256, H = 256;

    try {
        // 1. Initialise CUDA driver API.
        CU(cuInit(0));
        CUdevice cuDev;
        CU(cuDeviceGet(&cuDev, 0));
        CUcontext cuCtx;
        CU(cuCtxCreate(&cuCtx, 0, cuDev));

        // 2. Create headless Vulkan context.
        VulkanContext vk = createVulkan();

        // 3. Construct shared resources.
        interop::SharedImage    image(vk.device, vk.physDevice, W, H,
                                      VK_FORMAT_R8G8B8A8_UNORM);
        interop::SharedSemaphore sem(vk.device);

        // 4. Create a CUDA surface object over the shared CUarray.
        CUDA_RESOURCE_DESC resDesc{};
        resDesc.resType         = CU_RESOURCE_TYPE_ARRAY;
        resDesc.res.array.hArray = image.cuArray();
        CUsurfObject surf = 0;
        CU(cuSurfObjectCreate(&surf, &resDesc));

        // 5. Launch the fill kernel.
        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);
        fill_pattern<<<grid, block>>>(static_cast<cudaSurfaceObject_t>(surf), W, H);

        // 6. Signal the shared semaphore from CUDA (Vulkan will wait on it).
        sem.signal(0);  // stream 0

        CU(cuSurfObjectDestroy(surf));

        // 7. Allocate host-visible readback buffer.
        VkDeviceSize bufSize = W * H * 4;
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size  = bufSize;
        bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer readbackBuf;
        VK(vkCreateBuffer(vk.device, &bufInfo, nullptr, &readbackBuf));

        VkMemoryRequirements bufMemReqs;
        vkGetBufferMemoryRequirements(vk.device, readbackBuf, &bufMemReqs);
        uint32_t memIdx = findHostVisibleMemType(vk.physDevice, bufMemReqs.memoryTypeBits);

        VkMemoryAllocateInfo bufAlloc{};
        bufAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        bufAlloc.allocationSize  = bufMemReqs.size;
        bufAlloc.memoryTypeIndex = memIdx;
        VkDeviceMemory readbackMem;
        VK(vkAllocateMemory(vk.device, &bufAlloc, nullptr, &readbackMem));
        VK(vkBindBufferMemory(vk.device, readbackBuf, readbackMem, 0));

        // 8. Record: wait on sem, transition image, copy to buffer.
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK(vkBeginCommandBuffer(vk.cmdBuf, &beginInfo));

        // Transition UNDEFINED → TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask       = 0;
        barrier.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image.vkImage();
        barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(vk.cmdBuf,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        // Copy image → buffer
        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset       = {0, 0, 0};
        region.imageExtent       = {(uint32_t)W, (uint32_t)H, 1};
        vkCmdCopyImageToBuffer(vk.cmdBuf, image.vkImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               readbackBuf, 1, &region);

        VK(vkEndCommandBuffer(vk.cmdBuf));

        // Submit: wait for CUDA semaphore signal before executing the copy.
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSemaphore          waitSem   = sem.vkSemaphore();
        VkSubmitInfo submitInfo{};
        submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount   = 1;
        submitInfo.pWaitSemaphores      = &waitSem;
        submitInfo.pWaitDstStageMask    = &waitStage;
        submitInfo.commandBufferCount   = 1;
        submitInfo.pCommandBuffers      = &vk.cmdBuf;
        VK(vkQueueSubmit(vk.queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK(vkQueueWaitIdle(vk.queue));

        // 9. Read back and verify.
        void* mapped = nullptr;
        VK(vkMapMemory(vk.device, readbackMem, 0, bufSize, 0, &mapped));
        auto* pixels = reinterpret_cast<unsigned char*>(mapped);

        bool pass = true;
        for (int y = 0; y < H && pass; ++y) {
            for (int x = 0; x < W && pass; ++x) {
                int   idx      = (y * W + x) * 4;
                uint8_t expR   = static_cast<uint8_t>((x + y * W) % 256);
                uint8_t gotR   = pixels[idx + 0];
                if (gotR != expR) {
                    printf("[FAIL] Pixel (%d,%d): expected R=%u, got R=%u\n",
                           x, y, expR, gotR);
                    pass = false;
                }
            }
        }
        vkUnmapMemory(vk.device, readbackMem);

        // 10. Cleanup readback resources.
        vkFreeMemory(vk.device, readbackMem, nullptr);
        vkDestroyBuffer(vk.device, readbackBuf, nullptr);
        destroyVulkan(vk);
        cuCtxDestroy(cuCtx);

        if (pass) {
            printf("[PASS] Vulkan/CUDA interop verified (%dx%d RGBA, pattern round-trip)\n", W, H);
            return 0;
        } else {
            return 1;
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "[FAIL] Exception: %s\n", e.what());
        return 1;
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add interop-test/main.cpp
git commit -m "feat: add interop-test harness (headless Vulkan + CUDA round-trip)"
```

---

## Task 9: Build and run the test

**Files:** (build + run only)

- [ ] **Step 1: Build the full solution (layer + interop-test)**

```
powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' SMOOTHING-OPENXR-LAYER.sln /p:Configuration=Debug /p:Platform=x64 /p:RestorePackages=true /m"
```

Expected: `Build succeeded.` 0 errors. Both projects build. Common first-time errors and fixes:
- `CUDA 13.2.props not found` — CUDA 13.2 Toolkit is not installed or its MSBuild integration was not registered. Re-run the CUDA installer and select "Visual Studio Integration".
- `cuSurfObjectCreate` undefined — add `cudart.lib` to interop-test link dependencies (it should already be there per the vcxproj above).
- `fill_pattern` linker error — ensure `fill_pattern.cu` is listed as `CudaCompile` not `ClCompile` in the vcxproj.

- [ ] **Step 2: Run the test**

```
powershell.exe -Command "& '.\bin\x64\Debug\interop-test.exe'"
```

Expected output:
```
Using GPU: NVIDIA GeForce RTX 5070 Ti
[PASS] Vulkan/CUDA interop verified (256x256 RGBA, pattern round-trip)
```

Exit code: `0`

If `[FAIL] Pixel (0,0): expected R=0, got R=0` (all zeros) — the kernel ran but the image content was not transferred. Check that `CUDA_ARRAY3D_SURFACE_LDST` is set on the array descriptor in `SharedImage` constructor and that the semaphore signal/wait ordering is correct.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "feat: interop-test passes — Vulkan/CUDA SharedImage round-trip verified"
```

---

## Self-Review

**Spec coverage:**
- ✅ `SharedImage` RAII type with Vulkan + CUDA sides
- ✅ `SharedSemaphore` RAII type with signal/wait helpers
- ✅ Move-only semantics on both types
- ✅ Constructor-throws-on-failure, no two-phase init
- ✅ CUDA driver API in the module (`cuda.lib`), CUDA runtime only in test (`cudart.lib`)
- ✅ Test: headless Vulkan, CUDA kernel writes, semaphore sync, Vulkan readback, pixel verify
- ✅ Build system: CUDA paths added to layer vcxproj, test project with CUDA MSBuild integration
- ✅ Interop module compiled by both projects via shared file reference

**Type consistency check:**
- `SharedImage::vkImage()` → used as `image.vkImage()` in `main.cpp` ✅
- `SharedImage::cuArray()` → used in `CUDA_RESOURCE_DESC.res.array.hArray` in `main.cpp` ✅
- `SharedSemaphore::vkSemaphore()` → used in `VkSubmitInfo::pWaitSemaphores` in `main.cpp` ✅
- `SharedSemaphore::signal(CUstream)` → called as `sem.signal(0)` in `main.cpp` ✅
- `fill_pattern` kernel signature matches declaration in `main.cpp` ✅
