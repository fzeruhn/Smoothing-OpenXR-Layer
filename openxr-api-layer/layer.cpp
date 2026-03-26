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

#include "pch.h"
#include "layer.h"
#include <log.h>
#include <util.h>
#include <algorithm>
#include <deque>

#define VK_USE_PLATFORM_WIN32_KHR
#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr_platform.h>
#include <map>       // For tracking image indices

namespace openxr_api_layer {

    using namespace log;

    // Our API layer implement these extensions, and their specified version.
    const std::vector<std::pair<std::string, uint32_t>> advertisedExtensions = {};

    // Initialize these vectors with arrays of extensions to block and implicitly request for the instance.
    const std::vector<std::string> blockedExtensions = {};
    const std::vector<std::string> implicitExtensions = {};

    class VulkanFrameProcessor {
      public:
        VulkanFrameProcessor(VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue, uint32_t queueFamilyIndex)
            : m_physicalDevice(physicalDevice), m_device(device), m_queue(queue) {
            // Create Command Pool for compute/graphics operations
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = queueFamilyIndex;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool);

            // Allocate Command Buffer
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
            vkAllocateCommandBuffers(m_device, &allocInfo, &m_commandBuffer);

            // TODO: Initialize your Vulkan Compute Pipeline here
            // (Create Descriptor Sets, Pipeline Layout, Compute Pipeline for Motion Vectors + Warp)
        }

        ~VulkanFrameProcessor() {
            if (m_device) {
                vkDestroyCommandPool(m_device, m_commandPool, nullptr);
                // TODO: Destroy your compute pipelines here
            }
        }

        // ASYNCHRONOUS processing. No vkQueueWaitIdle!
        void ProcessFrames(VkImage colorCurrent, VkImage depthCurrent, VkImage colorPrevious, VkImage depthPrevious) {
            if (!colorPrevious || !depthPrevious)
                return; // Need two frames to do motion vectors

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

            // 1. Image Memory Barriers (Transition layouts to be readable by your compute shader)
            // ... (Insert vkCmdPipelineBarrier here to transition to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

            // 2. Bind your Compute Pipeline
            // vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);

            // 3. Bind Descriptor Sets (Passing in Current & Previous Color/Depth)
            // vkCmdBindDescriptorSets(...)

            // 4. Dispatch the compute shader
            // uint32_t groupCountX = (width + 15) / 16;
            // uint32_t groupCountY = (height + 15) / 16;
            // vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, 1);

            // 5. Image Memory Barriers (Transition back to COLOR_ATTACHMENT_OPTIMAL for OpenXR)
            // ...

            vkEndCommandBuffer(m_commandBuffer);

            // Submit work to the GPU queue. We do NOT wait for it to finish here.
            // OpenXR / the VR runtime will handle synchronization via Vulkan semaphores/fences natively.
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &m_commandBuffer;
            vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        }

      private:
        VkPhysicalDevice m_physicalDevice;
        VkDevice m_device;
        VkQueue m_queue;
        VkCommandPool m_commandPool;
        VkCommandBuffer m_commandBuffer;
        // TODO: Add VkPipeline, VkPipelineLayout, VkDescriptorSet etc.
    };

    // This class implements our API layer.
    class OpenXrLayer : public openxr_api_layer::OpenXrApi {
      public:
        OpenXrLayer() = default;
        ~OpenXrLayer() = default;

        // https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrGetInstanceProcAddr
        XrResult xrGetInstanceProcAddr(XrInstance instance, const char* name, PFN_xrVoidFunction* function) override {
            TraceLoggingWrite(g_traceProvider,
                              "xrGetInstanceProcAddr",
                              TLXArg(instance, "Instance"),
                              TLArg(name, "Name"),
                              TLArg(m_bypassApiLayer, "Bypass"));

            XrResult result = m_bypassApiLayer ? m_xrGetInstanceProcAddr(instance, name, function)
                                               : OpenXrApi::xrGetInstanceProcAddr(instance, name, function);

            TraceLoggingWrite(g_traceProvider, "xrGetInstanceProcAddr", TLPArg(*function, "Function"));

            return result;
        }

        // https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrCreateInstance
        XrResult xrCreateInstance(const XrInstanceCreateInfo* createInfo) override {
            if (createInfo->type != XR_TYPE_INSTANCE_CREATE_INFO) {
                return XR_ERROR_VALIDATION_FAILURE;
            }

            // Needed to resolve the requested function pointers.
            OpenXrApi::xrCreateInstance(createInfo);

            // Dump the application name, OpenXR runtime information and other useful things for debugging.
            TraceLoggingWrite(g_traceProvider,
                              "xrCreateInstance",
                              TLArg(xr::ToString(createInfo->applicationInfo.apiVersion).c_str(), "ApiVersion"),
                              TLArg(createInfo->applicationInfo.applicationName, "ApplicationName"),
                              TLArg(createInfo->applicationInfo.applicationVersion, "ApplicationVersion"),
                              TLArg(createInfo->applicationInfo.engineName, "EngineName"),
                              TLArg(createInfo->applicationInfo.engineVersion, "EngineVersion"),
                              TLArg(createInfo->createFlags, "CreateFlags"));
            Log(fmt::format("Application: {}\n", createInfo->applicationInfo.applicationName));

            // Here there can be rules to disable the API layer entirely (based on applicationName for example).
            // m_bypassApiLayer = ...

            if (m_bypassApiLayer) {
                Log(fmt::format("{} layer will be bypassed\n", LayerName));
                return XR_SUCCESS;
            }

            for (uint32_t i = 0; i < createInfo->enabledApiLayerCount; i++) {
                TraceLoggingWrite(
                    g_traceProvider, "xrCreateInstance", TLArg(createInfo->enabledApiLayerNames[i], "ApiLayerName"));
            }
            for (uint32_t i = 0; i < createInfo->enabledExtensionCount; i++) {
                TraceLoggingWrite(
                    g_traceProvider, "xrCreateInstance", TLArg(createInfo->enabledExtensionNames[i], "ExtensionName"));
            }

            XrInstanceProperties instanceProperties = {XR_TYPE_INSTANCE_PROPERTIES};
            CHECK_XRCMD(OpenXrApi::xrGetInstanceProperties(GetXrInstance(), &instanceProperties));
            const auto runtimeName = fmt::format("{} {}.{}.{}",
                                                 instanceProperties.runtimeName,
                                                 XR_VERSION_MAJOR(instanceProperties.runtimeVersion),
                                                 XR_VERSION_MINOR(instanceProperties.runtimeVersion),
                                                 XR_VERSION_PATCH(instanceProperties.runtimeVersion));
            TraceLoggingWrite(g_traceProvider, "xrCreateInstance", TLArg(runtimeName.c_str(), "RuntimeName"));
            Log(fmt::format("Using OpenXR runtime: {}\n", runtimeName));

            return XR_SUCCESS;
        }

        // https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrGetSystem
        XrResult xrGetSystem(XrInstance instance, const XrSystemGetInfo* getInfo, XrSystemId* systemId) override {
            if (getInfo->type != XR_TYPE_SYSTEM_GET_INFO) {
                return XR_ERROR_VALIDATION_FAILURE;
            }

            TraceLoggingWrite(g_traceProvider,
                              "xrGetSystem",
                              TLXArg(instance, "Instance"),
                              TLArg(xr::ToCString(getInfo->formFactor), "FormFactor"));

            const XrResult result = OpenXrApi::xrGetSystem(instance, getInfo, systemId);
            if (XR_SUCCEEDED(result) && getInfo->formFactor == XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY) {
                if (*systemId != m_systemId) {
                    XrSystemProperties systemProperties{XR_TYPE_SYSTEM_PROPERTIES};
                    CHECK_XRCMD(OpenXrApi::xrGetSystemProperties(instance, *systemId, &systemProperties));
                    TraceLoggingWrite(g_traceProvider, "xrGetSystem", TLArg(systemProperties.systemName, "SystemName"));
                    Log(fmt::format("Using OpenXR system: {}\n", systemProperties.systemName));
                }

                // Remember the XrSystemId to use.
                m_systemId = *systemId;
            }

            TraceLoggingWrite(g_traceProvider, "xrGetSystem", TLArg((int)*systemId, "SystemId"));

            return result;
        }

        // https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrCreateSession
        // 1. HOOK SESSION CREATION
        XrResult xrCreateSession(XrInstance instance,
                                 const XrSessionCreateInfo* createInfo,
                                 XrSession* session) override {
            if (createInfo->type != XR_TYPE_SESSION_CREATE_INFO) {
                return XR_ERROR_VALIDATION_FAILURE;
            }

            // Check what engine the game is using
            const XrBaseInStructure* entry = reinterpret_cast<const XrBaseInStructure*>(createInfo->next);
            while (entry) {
                if (entry->type == XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR) {
                    Log("Star Citizen is using VULKAN\n");
                    m_apiType = GraphicsAPI::Vulkan;

                    // Capture the Vulkan handles the game is using
                    const auto* vkBinding = reinterpret_cast<const XrGraphicsBindingVulkanKHR*>(entry);
                    m_vkInstance = vkBinding->instance;
                    m_vkPhysicalDevice = vkBinding->physicalDevice;
                    m_vkDevice = vkBinding->device;
                    m_vkQueueFamilyIndex = vkBinding->queueFamilyIndex;
                } else if (entry->type == XR_TYPE_GRAPHICS_BINDING_D3D11_KHR) {
                    Log("Game is using DX11\n");
                    m_apiType = GraphicsAPI::DX11;
                }
                entry = entry->next;
            }

            XrResult result = OpenXrApi::xrCreateSession(instance, createInfo, session);

            if (XR_SUCCEEDED(result) && isSystemHandled(createInfo->systemId)) {
                m_session = *session;
                Log(fmt::format("Captured VR Session: {}\n", (void*)m_session));
            }
            return result;
        }

        // 2. HOOK SWAPCHAIN CREATION
        XrResult xrCreateSwapchain(XrSession session,
                                   const XrSwapchainCreateInfo* createInfo,
                                   XrSwapchain* swapchain) override {
            XrResult result = OpenXrApi::xrCreateSwapchain(session, createInfo, swapchain);

            if (XR_SUCCEEDED(result) && session == m_session && m_apiType == GraphicsAPI::Vulkan) {
                // Get color
                if (createInfo->usageFlags & XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT) {
                    Log(fmt::format("Intercepted Color Swapchain Creation! Width: {} Height: {}\n", createInfo->width, createInfo->height));
                    // Save this swapchain handle so we know to track its images
                    m_colorSwapchains.push_back(*swapchain);

                    // Initialize the processor
                    if (!m_processor && m_vkDevice != VK_NULL_HANDLE) {
                        VkQueue queue;
                        vkGetDeviceQueue(m_vkDevice, m_vkQueueFamilyIndex, 0, &queue);
                        m_processor = std::make_unique<VulkanFrameProcessor>(
                            m_vkPhysicalDevice, m_vkDevice, queue, m_vkQueueFamilyIndex);
                        Log("Vulkan Frame Processor Initialized!\n");
                    }

                // Get depth
                } else if (createInfo->usageFlags & XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                    Log(fmt::format("Intercepted Depth Swapchain! Width: {} Height: {}\n", createInfo->width, createInfo->height));
                    m_depthSwapchains.push_back(*swapchain);
                }
            }
            return result;
        }

        // 3. HOOK SWAPCHAIN IMAGES (The actual GPU Textures)
        XrResult xrEnumerateSwapchainImages(XrSwapchain swapchain,
                                            uint32_t imageCapacityInput,
                                            uint32_t* imageCountOutput,
                                            XrSwapchainImageBaseHeader* images) override {
            XrResult result =
                OpenXrApi::xrEnumerateSwapchainImages(swapchain, imageCapacityInput, imageCountOutput, images);

            if (XR_SUCCEEDED(result) && images != nullptr && m_apiType == GraphicsAPI::Vulkan) {
                // If this is a color or depth swapchain we care about...
                bool isTracked =
                    std::find(m_colorSwapchains.begin(), m_colorSwapchains.end(), swapchain) !=
                        m_colorSwapchains.end() ||
                    std::find(m_depthSwapchains.begin(), m_depthSwapchains.end(), swapchain) != m_depthSwapchains.end();

                if (isTracked) {
                    XrSwapchainImageVulkanKHR* vkImages = reinterpret_cast<XrSwapchainImageVulkanKHR*>(images);
                    for (uint32_t i = 0; i < *imageCountOutput; i++) {
                        m_vulkanImages[swapchain].push_back(vkImages[i].image);
                    }
                    Log(fmt::format("Mapped {} Vulkan images for swapchain.\n", *imageCountOutput));
                }
            }
            return result;
        }

        // HOOK ACQUIRE IMAGE (Tells us which texture index in the swapchain is active)
        XrResult xrAcquireSwapchainImage(XrSwapchain swapchain,
                                         const XrSwapchainImageAcquireInfo* acquireInfo,
                                         uint32_t* index) override {
            XrResult result = OpenXrApi::xrAcquireSwapchainImage(swapchain, acquireInfo, index);
            if (XR_SUCCEEDED(result)) {
                m_acquiredIndices[swapchain] = *index; // Remember this index for EndFrame
            }
            return result;
        }

        // 4. HOOK END FRAME (The trigger to copy the data)
        XrResult xrEndFrame(XrSession session, const XrFrameEndInfo* frameEndInfo) override {
            if (session == m_session && m_apiType == GraphicsAPI::Vulkan) {
                VkImage currentColor = VK_NULL_HANDLE;
                VkImage currentDepth = VK_NULL_HANDLE;

                // 1. Grab Current Color
                if (!m_colorSwapchains.empty()) {
                    XrSwapchain colorChain = m_colorSwapchains[0];
                    currentColor = m_vulkanImages[colorChain][m_acquiredIndices[colorChain]];
                }

                // 2. Grab Current Depth
                if (!m_depthSwapchains.empty()) {
                    XrSwapchain depthChain = m_depthSwapchains[0];
                    currentDepth = m_vulkanImages[depthChain][m_acquiredIndices[depthChain]];
                }

                // 3. Process if we have a previous frame!
                if (m_processor && currentColor && currentDepth && m_prevColor && m_prevDepth) {
                    // Dispatch to GPU. Do NOT wait for idle.
                    m_processor->ProcessFrames(currentColor, currentDepth, m_prevColor, m_prevDepth);
                }

                // 4. Update our history buffers for the NEXT frame (t-1)
                m_prevColor = currentColor;
                m_prevDepth = currentDepth;
            }

            // Finally, pass it back to the OpenXR runtime so it gets to the headset
            return OpenXrApi::xrEndFrame(session, frameEndInfo);
        }


      private:
        bool isSystemHandled(XrSystemId systemId) const {return systemId == m_systemId;}
        enum class GraphicsAPI { Unknown, DX11, Vulkan };
        GraphicsAPI m_apiType{GraphicsAPI::Unknown};
        bool m_bypassApiLayer{false};
        XrSystemId m_systemId{XR_NULL_SYSTEM_ID};
        XrSession m_session{XR_NULL_HANDLE};

        std::unique_ptr<VulkanFrameProcessor> m_processor;

        // Vulkan Tracking
        VkInstance m_vkInstance{VK_NULL_HANDLE};
        VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};
        VkDevice m_vkDevice{VK_NULL_HANDLE};
        uint32_t m_vkQueueFamilyIndex{0};

        // Frame Tracking
        std::vector<XrSwapchain> m_colorSwapchains;
        std::vector<XrSwapchain> m_depthSwapchains;
        std::map<XrSwapchain, std::vector<VkImage>> m_vulkanImages;
        std::map<XrSwapchain, uint32_t> m_acquiredIndices;

        // NEW: History tracking for Frame Generation/Warping
        VkImage m_prevColor{VK_NULL_HANDLE};
        VkImage m_prevDepth{VK_NULL_HANDLE};
    };

    // This method is required by the framework to instantiate your OpenXrApi implementation.
    OpenXrApi* GetInstance() {
        if (!g_instance) {
            g_instance = std::make_unique<OpenXrLayer>();
        }
        return g_instance.get();
    }

} // namespace openxr_api_layer

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        TraceLoggingRegister(openxr_api_layer::log::g_traceProvider);
        break;

    case DLL_PROCESS_DETACH:
        TraceLoggingUnregister(openxr_api_layer::log::g_traceProvider);
        break;

    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    }
    return TRUE;
}
