// MIT License
//
// << insert your own copyright here >>
//
// Based on https://github.com/mbucchia/OpenXR-Layer-Template.
// Copyright(c) 2022-2023 Matthieu Bucchianeri
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright noticeand this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// 1. Target API Definitions
#define XR_USE_GRAPHICS_API_D3D11
#define XR_USE_GRAPHICS_API_D3D12
#define XR_USE_GRAPHICS_API_VULKAN
#define VK_USE_PLATFORM_WIN32_KHR
#define XR_USE_PLATFORM_WIN32
#define XR_NO_PROTOTYPES

// 2. Standard library
#include <algorithm>
#include <cstdarg>
#include <ctime>
#define _USE_MATH_DEFINES
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <optional>
using namespace std::chrono_literals;

// 3. Windows header files
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <unknwn.h>
#include <wrl.h>
#include <wil/resource.h>
#include <traceloggingactivity.h>
#include <traceloggingprovider.h>
using Microsoft::WRL::ComPtr;

// 4. Graphics APIs (MUST be included before OpenXR)
#include <dxgiformat.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <vulkan/vulkan.h>

// 5. OpenXR
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <loader_interfaces.h>

// 6. OpenXR/DirectX utilities
#include <XrError.h>
#include <XrMath.h>
#include <XrSide.h>
#include <XrStereoView.h>
#include <XrToString.h>
#include <DirectXCollision.h>

// 7. FMT formatter & Utilities
#include <fmt/format.h>
#include <utils/graphics.h>
#include <utils/inputs.h>
