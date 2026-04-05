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
#include "depth_provider.h"
#include "frame_broker.h"
#include "frame_context.h"
#include "pose_provider.h"
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

// TODO (Item 4): Uncomment when pose warp is fully integrated
// #include "pose_warp.h"
// #include "pose_warp_math.h"


namespace openxr_api_layer {

    using namespace log;

    // Vulkan result check for internal layer use (not to be confused with CHECK_XRCMD).
    // Throws on failure so that constructor exceptions are visible and the layer doesn't
    // silently proceed with a null command pool / buffer.
#define CHECK_VK_LAYER(call) \
    do { \
        VkResult _vr = (call); \
        if (_vr != VK_SUCCESS) \
            throw std::runtime_error("Vulkan error " + std::to_string(static_cast<int>(_vr)) \
                                     + " in " #call); \
    } while (0)

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
            CHECK_VK_LAYER(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool));

            // Allocate Command Buffer
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
            CHECK_VK_LAYER(vkAllocateCommandBuffers(m_device, &allocInfo, &m_commandBuffer));

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
            CHECK_VK_LAYER(vkBeginCommandBuffer(m_commandBuffer, &beginInfo));

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

            CHECK_VK_LAYER(vkEndCommandBuffer(m_commandBuffer));

            // Submit work to the GPU queue. We do NOT wait for it to finish here.
            // OpenXR / the VR runtime will handle synchronization via Vulkan semaphores/fences natively.
            //
            // TODO (Item 10 — Frame Submission): This stub reuses a single command buffer with no
            // fence.  Once ProcessFrames() records real GPU work, submitting without a fence while
            // the previous submit may still be in flight is a Vulkan validation error.  Replace with
            // either: (a) per-frame fences + CPU wait before re-recording, or (b) a ring of N
            // command buffers (N >= 2 for double-buffering) so the CPU never races the GPU.
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &m_commandBuffer;
            CHECK_VK_LAYER(vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE));
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
                    const uint32_t previousWidth = m_frameBroker.GetSwapchainWidth();
                    const uint32_t previousHeight = m_frameBroker.GetSwapchainHeight();
                    m_frameBroker.RegisterSwapchain(*swapchain, *createInfo);
                    Log(fmt::format("Intercepted Color Swapchain Creation! Width: {} Height: {}\n", createInfo->width, createInfo->height));

                    // Store swapchain resolution for future stereo-adapter wiring.
                    if (previousWidth != 0 && previousHeight != 0 &&
                        (previousWidth != createInfo->width || previousHeight != createInfo->height)) {
                        Log(fmt::format("[WARN] Color swapchain resolution changed from {}x{} to {}x{}; "
                            "future stereo adaptation must use per-swapchain dimensions\n",
                            previousWidth, previousHeight, createInfo->width, createInfo->height));
                    }

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
                    m_frameBroker.RegisterSwapchain(*swapchain, *createInfo);
                    Log(fmt::format("Intercepted Depth Swapchain! Width: {} Height: {}\n", createInfo->width, createInfo->height));
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
                if (m_frameBroker.IsColorSwapchain(swapchain) || m_frameBroker.IsDepthSwapchain(swapchain)) {
                    m_frameBroker.RegisterSwapchainImages(swapchain, *imageCountOutput, images);
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
                m_frameBroker.OnAcquireSwapchainImage(swapchain, *index); // Remember this index for EndFrame
            }
            return result;
        }

        XrResult xrWaitFrame(XrSession session, const XrFrameWaitInfo* frameWaitInfo, XrFrameState* frameState) override {
            XrResult result = OpenXrApi::xrWaitFrame(session, frameWaitInfo, frameState);
            if (XR_SUCCEEDED(result) && session == m_session && frameState && frameState->type == XR_TYPE_FRAME_STATE) {
                m_poseProvider.OnWaitFrame(*frameState);
            }
            return result;
        }

        // 4. HOOK END FRAME (The trigger to copy the data)
        XrResult xrEndFrame(XrSession session, const XrFrameEndInfo* frameEndInfo) override {
            if (session == m_session && m_apiType == GraphicsAPI::Vulkan) {
                m_frameContext = FrameContext{};
                const XrCompositionLayerProjection* projectionLayer = nullptr;

                // Extract per-eye FOV from projection layers
                if (frameEndInfo && frameEndInfo->layerCount > 0) {
                    for (uint32_t i = 0; i < frameEndInfo->layerCount; ++i) {
                        const XrCompositionLayerBaseHeader* layer = frameEndInfo->layers[i];
                        if (layer && layer->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION) {
                            projectionLayer =
                                reinterpret_cast<const XrCompositionLayerProjection*>(layer);
                            
                            // Extract FOV from both eyes
                            if (projectionLayer->viewCount >= 2) {
                                m_fovLeft = projectionLayer->views[0].fov;
                                m_fovRight = projectionLayer->views[1].fov;
                                for (uint32_t eye = 0; eye < 2; ++eye) {
                                    m_frameContext.renderViews[eye].fov = projectionLayer->views[eye].fov;
                                    m_frameContext.renderViews[eye].pose = projectionLayer->views[eye].pose;
                                    m_frameContext.renderViews[eye].valid = true;
                                }
                                
                                if (!m_fovInitialized) {
                                    m_fovInitialized = true;
                                    TraceLoggingWrite(g_traceProvider,
                                                      "FOV_Extracted",
                                                      TLArg(m_fovLeft.angleLeft, "LeftEye_AngleLeft"),
                                                      TLArg(m_fovLeft.angleRight, "LeftEye_AngleRight"),
                                                      TLArg(m_fovLeft.angleUp, "LeftEye_AngleUp"),
                                                      TLArg(m_fovLeft.angleDown, "LeftEye_AngleDown"),
                                                      TLArg(m_fovRight.angleLeft, "RightEye_AngleLeft"),
                                                      TLArg(m_fovRight.angleRight, "RightEye_AngleRight"),
                                                      TLArg(m_fovRight.angleUp, "RightEye_AngleUp"),
                                                      TLArg(m_fovRight.angleDown, "RightEye_AngleDown"));
                                }

                                // StereoVectorAdapter wiring is compiled in tests first.
                                // Runtime integration into the layer remains pending.
                            }
                            break; // Only process first projection layer
                        }
                    }
                }

                if (projectionLayer) {
                    m_depthProvider.SetSwapchainImageLookup(m_frameBroker.GetVulkanImages(), m_frameBroker.GetAcquiredIndices());
                    const XrResult poseResult =
                        m_poseProvider.PopulatePredictedViews(*this, session, *projectionLayer, m_frameContext);
                    if (XR_FAILED(poseResult)) {
                        TraceLoggingWrite(g_traceProvider,
                                          "PoseProvider_Failed",
                                          TLArg(xr::ToCString(poseResult), "Result"));
                    }

                    m_depthProvider.ExtractDepthInfo(*projectionLayer, m_frameContext);
                    const bool hasDepthChain = m_frameContext.depthViews[0].valid || m_frameContext.depthViews[1].valid;
                    if (!hasDepthChain && !m_depthWarningLogged) {
                        m_depthWarningLogged = true;
                        Log("[WARN] No XR_KHR_composition_layer_depth info on projection views; using tracked depth swapchains.\n");
                    }

                    if (hasDepthChain) {
                        const auto& depthView = m_frameContext.depthViews[0].valid ? m_frameContext.depthViews[0] : m_frameContext.depthViews[1];
                        TraceLoggingWrite(g_traceProvider,
                                          "Depth_Metadata",
                                          TLArg(depthView.minDepth, "MinDepth"),
                                          TLArg(depthView.maxDepth, "MaxDepth"),
                                          TLArg(depthView.nearZ, "NearZ"),
                                          TLArg(depthView.farZ, "FarZ"),
                                          TLArg(depthView.reversedZ, "ReversedZ"));
                    }
                }

                VkImage currentColor = m_frameBroker.GetCurrentColorImage();
                VkImage currentDepth = VK_NULL_HANDLE;

                // 2. Grab Current Depth (prefer depth chained on projection views)
                if (m_frameContext.depthViews[0].valid) {
                    currentDepth = m_frameContext.depthViews[0].image;
                } else if (m_frameContext.depthViews[1].valid) {
                    currentDepth = m_frameContext.depthViews[1].image;
                } else {
                    currentDepth = m_frameBroker.GetCurrentDepthImage();
                }

                // 3. Process if we have a previous frame!
                if (m_processor && currentColor && currentDepth && m_prevColor && m_prevDepth) {
                    // TODO (Item 4): Pre-OFA pose pre-warp
                    // When Item 2 (Pose Data Pipeline) is complete:
                    //   1. Extract XrPosef render_pose and display_pose from frameEndInfo projection layers
                    //   2. Compute pose_delta = display_pose * inverse(render_pose)
                    //   3. Extract rotation quaternion from pose_delta
                    //   4. Compute homography from rotation and m_fovLeft (using pose_warp_math::computeIntrinsics)
                    //   5. Create temporary CUarrays from m_prevColor via Vulkan/CUDA interop
                    //   6. Call m_poseWarper->warp() to apply homography to m_prevColor
                    //   7. Feed warped frame to OFA (when OFA integration is complete)
                    // This removes camera rotation from the motion field, improving OFA quality.

                    // Dispatch to GPU. Do NOT wait for idle.
                    m_processor->ProcessFrames(currentColor, currentDepth, m_prevColor, m_prevDepth);
                    m_synthesizedColor = currentColor;
                    m_hasSynthesisOutput = true;
                }

                // TODO: When OFA pipeline is integrated, invoke stereo vector adaptation here.
                // Note: current StereoVectorAdapter::adapt() performs cudaDeviceSynchronize();
                // hot-path integration should use a stream-based API to avoid CPU blocking.

                // 4. Update our history buffers for the NEXT frame (t-1)
                m_prevColor = currentColor;
                m_prevDepth = currentDepth;

                TraceLoggingWrite(g_traceProvider,
                                  "Frame_Broker_State",
                                  TLArg(currentColor != VK_NULL_HANDLE, "HasColor"),
                                  TLArg(currentDepth != VK_NULL_HANDLE, "HasDepth"),
                                  TLArg(m_hasSynthesisOutput, "HasSynthesisOutput"));
            }

            // Item 10 wiring stage: keep runtime submission unchanged for now while reporting
            // synthesis readiness. Actual layer replacement/injection is next step.
            if (m_hasSynthesisOutput && m_synthesizedColor != VK_NULL_HANDLE) {
                TraceLoggingWrite(g_traceProvider, "Frame_Injection_Ready", TLArg(true, "Ready"));
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
        FrameBroker m_frameBroker;
        PoseProvider m_poseProvider;
        DepthProvider m_depthProvider;
        FrameContext m_frameContext{};
        bool m_depthWarningLogged{false};
        VkImage m_synthesizedColor{VK_NULL_HANDLE};
        bool m_hasSynthesisOutput{false};

        // Vulkan Tracking
        VkInstance m_vkInstance{VK_NULL_HANDLE};
        VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};
        VkDevice m_vkDevice{VK_NULL_HANDLE};
        uint32_t m_vkQueueFamilyIndex{0};

        // NEW: History tracking for Frame Generation/Warping
        VkImage m_prevColor{VK_NULL_HANDLE};
        VkImage m_prevDepth{VK_NULL_HANDLE};

        // Per-eye FOV data (extracted from xrEndFrame projection layers)
        XrFovf m_fovLeft{};
        XrFovf m_fovRight{};
        bool m_fovInitialized{false};

        // Placeholder: near/far planes for depth linearization (TODO: extract from projection matrix)
        // These are typical VR values; should be extracted from actual projection in future
        static constexpr float NEAR_PLANE = 0.1f;
        static constexpr float FAR_PLANE = 100.0f;
        static constexpr float IPD = 0.063f; // 63mm typical; TODO: query from runtime API

        // TODO (Item 4): Pre-OFA pose pre-warp integration
        // When Item 2 (Pose Data Pipeline) is complete:
        //   1. Add std::unique_ptr<pose_warp::PoseWarper> m_poseWarper;
        //   2. Uncomment pose warp includes at top of file
        //   3. Extract pose delta in xrEndFrame (display_pose * inverse(render_pose))
        //   4. Compute homography from pose delta and FOV (m_fovLeft/m_fovRight)
        //   5. Call m_poseWarper->warp() on m_prevColor before feeding to OFA
        // This will significantly improve OFA quality by removing camera motion from flow field.
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
