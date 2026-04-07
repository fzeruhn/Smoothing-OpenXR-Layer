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
#include "frame_injection.h"
#include "holding_pen.h"
#include "pose_provider.h"
#include "runtime_thread.h"
#include <log.h>
#include <util.h>
#include <algorithm>
#include <array>

#define VK_USE_PLATFORM_WIN32_KHR
#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr_platform.h>
#include <map>
#include <vector>


namespace openxr_api_layer {

    using namespace log;

    // Vulkan result check for internal layer use (not to be confused with CHECK_XRCMD).
    // Returns from the enclosing bool function on failure.
#define CHECK_VK_LAYER(call) \
    do { \
        VkResult _vr = (call); \
        if (_vr != VK_SUCCESS) { \
            Log(fmt::format("[ERROR] Vulkan call failed: {} (VkResult={})\n", #call, static_cast<int>(_vr))); \
            return false; \
        } \
    } while (0)

    // Safety guard: assert that an image being used in a color barrier is not a known depth handle.
    // Catches the "D32 transitioned as color" bug from previous Phase 3 attempts.
#define ASSERT_COLOR_BARRIER(img, knownDepthImg) \
    do { \
        if ((img) != VK_NULL_HANDLE && (img) == (knownDepthImg)) { \
            Log(fmt::format("[FATAL] Color barrier issued on depth image handle {:p} in " __FUNCTION__ "\n", \
                            static_cast<void*>(img))); \
            return false; \
        } \
    } while (0)

    // Our API layer implement these extensions, and their specified version.
    const std::vector<std::pair<std::string, uint32_t>> advertisedExtensions = {};

    // Initialize these vectors with arrays of extensions to block and implicitly request for the instance.
    const std::vector<std::string> blockedExtensions = {};
    const std::vector<std::string> implicitExtensions = {};

    class OpenXrLayer;

    XrResult XRAPI_CALL xrGetVulkanInstanceExtensionsKHR_intercept(XrInstance instance,
                                                                    XrSystemId systemId,
                                                                    uint32_t bufferCapacityInput,
                                                                    uint32_t* bufferCountOutput,
                                                                    char* buffer);

    XrResult XRAPI_CALL xrGetVulkanDeviceExtensionsKHR_intercept(XrInstance instance,
                                                                  XrSystemId systemId,
                                                                  uint32_t bufferCapacityInput,
                                                                  uint32_t* bufferCountOutput,
                                                                  char* buffer);

    // This class implements our API layer.
    class OpenXrLayer : public openxr_api_layer::OpenXrApi {
      public:
        OpenXrLayer() = default;
        ~OpenXrLayer() = default;

        XrResult xrGetVulkanInstanceExtensionsKHR(XrInstance instance,
                                                  XrSystemId systemId,
                                                  uint32_t bufferCapacityInput,
                                                  uint32_t* bufferCountOutput,
                                                  char* buffer) {
            ResolveVulkanExtensionFunctions(instance);
            if (m_xrGetVulkanInstanceExtensionsKHR == nullptr) {
                return XR_ERROR_FUNCTION_UNSUPPORTED;
            }
            return m_xrGetVulkanInstanceExtensionsKHR(instance, systemId, bufferCapacityInput, bufferCountOutput, buffer);
        }

        XrResult xrGetVulkanDeviceExtensionsKHR(XrInstance instance,
                                                XrSystemId systemId,
                                                uint32_t bufferCapacityInput,
                                                uint32_t* bufferCountOutput,
                                                char* buffer) {
            ResolveVulkanExtensionFunctions(instance);
            if (m_xrGetVulkanDeviceExtensionsKHR == nullptr) {
                return XR_ERROR_FUNCTION_UNSUPPORTED;
            }

            uint32_t runtimeCount = 0;
            const XrResult queryResult = m_xrGetVulkanDeviceExtensionsKHR(instance, systemId, 0, &runtimeCount, nullptr);
            if (XR_FAILED(queryResult)) {
                return queryResult;
            }

            std::string runtimeExtensions;
            runtimeExtensions.resize(runtimeCount == 0 ? 0 : runtimeCount - 1);
            if (runtimeCount > 0) {
                std::vector<char> runtimeBuffer(runtimeCount);
                const XrResult fetchResult =
                    m_xrGetVulkanDeviceExtensionsKHR(instance, systemId, runtimeCount, &runtimeCount, runtimeBuffer.data());
                if (XR_FAILED(fetchResult)) {
                    return fetchResult;
                }
                runtimeExtensions.assign(runtimeBuffer.data());
            }

            // Phase 3: CUDA pipeline is inactive. No external memory/semaphore extensions needed.
            static const std::array<const char*, 0> requiredDeviceExtensions = {};

            std::istringstream extStream(runtimeExtensions);
            std::vector<std::string> extensions;
            std::string token;
            while (extStream >> token) {
                extensions.push_back(token);
            }

            auto hasExtension = [&](const char* extension) {
                return std::find(extensions.begin(), extensions.end(), extension) != extensions.end();
            };

            for (const char* extension : requiredDeviceExtensions) {
                if (!hasExtension(extension)) {
                    extensions.emplace_back(extension);
                }
            }

            std::string mergedExtensions;
            for (size_t i = 0; i < extensions.size(); ++i) {
                if (i != 0) {
                    mergedExtensions.push_back(' ');
                }
                mergedExtensions += extensions[i];
            }

            const uint32_t mergedCount = static_cast<uint32_t>(mergedExtensions.size()) + 1;
            if (bufferCountOutput != nullptr) {
                *bufferCountOutput = mergedCount;
            }

            TraceLoggingWrite(g_traceProvider,
                              "Vulkan_Device_Extensions",
                              TLArg(runtimeExtensions.c_str(), "RuntimeExtensions"),
                              TLArg(mergedExtensions.c_str(), "MergedExtensions"));

            if (bufferCapacityInput == 0 || buffer == nullptr) {
                return XR_SUCCESS;
            }

            if (bufferCapacityInput < mergedCount) {
                return XR_ERROR_SIZE_INSUFFICIENT;
            }

            strcpy_s(buffer, bufferCapacityInput, mergedExtensions.c_str());
            return XR_SUCCESS;
        }

        // https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#xrGetInstanceProcAddr
        XrResult xrGetInstanceProcAddr(XrInstance instance, const char* name, PFN_xrVoidFunction* function) override {
            TraceLoggingWrite(g_traceProvider,
                              "xrGetInstanceProcAddr",
                              TLXArg(instance, "Instance"),
                              TLArg(name, "Name"),
                              TLArg(m_bypassApiLayer, "Bypass"));

            if (!m_bypassApiLayer && name != nullptr && function != nullptr) {
                if (strcmp(name, "xrGetVulkanInstanceExtensionsKHR") == 0) {
                    *function = reinterpret_cast<PFN_xrVoidFunction>(xrGetVulkanInstanceExtensionsKHR_intercept);
                    return XR_SUCCESS;
                }
                if (strcmp(name, "xrGetVulkanDeviceExtensionsKHR") == 0) {
                    *function = reinterpret_cast<PFN_xrVoidFunction>(xrGetVulkanDeviceExtensionsKHR_intercept);
                    return XR_SUCCESS;
                }
            }

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
            ResolveVulkanExtensionFunctions(GetXrInstance());

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
                    m_vkQueueIndex = vkBinding->queueIndex;
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

                PFN_xrVoidFunction waitFn = nullptr;
                if (XR_SUCCEEDED(OpenXrApi::xrGetInstanceProcAddr(instance, "xrWaitSwapchainImage", &waitFn))) {
                    m_xrWaitSwapchainImage = reinterpret_cast<PFN_xrWaitSwapchainImage>(waitFn);
                }

                PFN_xrVoidFunction releaseFn = nullptr;
                if (XR_SUCCEEDED(OpenXrApi::xrGetInstanceProcAddr(instance, "xrReleaseSwapchainImage", &releaseFn))) {
                    m_xrReleaseSwapchainImage = reinterpret_cast<PFN_xrReleaseSwapchainImage>(releaseFn);
                }
            }
            return result;
        }

        // 2. HOOK SWAPCHAIN CREATION
        XrResult xrCreateSwapchain(XrSession session,
                                   const XrSwapchainCreateInfo* createInfo,
                                   XrSwapchain* swapchain) override {
            XrResult result = OpenXrApi::xrCreateSwapchain(session, createInfo, swapchain);

            if (XR_SUCCEEDED(result) && session == m_session && m_apiType == GraphicsAPI::Vulkan) {
                if (createInfo->usageFlags & XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT) {
                    m_frameBroker.RegisterSwapchain(*swapchain, *createInfo);
                    Log(fmt::format("Intercepted Color Swapchain: {}x{}\n", createInfo->width, createInfo->height));

                    // Capture queue once on first color swapchain
                    if (m_vkQueue == VK_NULL_HANDLE && m_vkDevice != VK_NULL_HANDLE) {
                        vkGetDeviceQueue(m_vkDevice, m_vkQueueFamilyIndex, m_vkQueueIndex, &m_vkQueue);
                    }

                    // Initialize holding pen once (use the app swapchain format/dimensions)
                    if (!m_holdingPen.IsInitialized() && m_vkDevice != VK_NULL_HANDLE && m_vkQueue != VK_NULL_HANDLE) {
                        const VkFormat penFormat = static_cast<VkFormat>(createInfo->format);
                        if (m_holdingPen.Initialize(m_vkDevice, m_vkPhysicalDevice,
                                                    createInfo->width, createInfo->height,
                                                    penFormat, m_vkQueueFamilyIndex)) {
                            Log(fmt::format("[HoldingPen] Ready for {}x{} fmt={}\n",
                                            createInfo->width, createInfo->height,
                                            static_cast<int>(penFormat)));
                        } else {
                            Log("[HoldingPen] Initialization failed\n");
                        }
                    }

                    // Ensure injection swapchain exists (guard prevents re-entrant creation)
                    if (!FrameInjection::IsCreatingSwapchain()) {
                        m_frameInjection.EnsureSwapchain(*this, session, m_frameBroker);
                        if (m_frameInjection.IsReady() && m_injectionVulkanImages.empty()) {
                            uint32_t count = 0;
                            if (XR_SUCCEEDED(OpenXrApi::xrEnumerateSwapchainImages(
                                    m_frameInjection.Swapchain(), 0, &count, nullptr)) && count > 0) {
                                std::vector<XrSwapchainImageVulkanKHR> imgs(
                                    count, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
                                if (XR_SUCCEEDED(OpenXrApi::xrEnumerateSwapchainImages(
                                        m_frameInjection.Swapchain(), count, &count,
                                        reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data())))) {
                                    for (auto& img : imgs) {
                                        m_injectionVulkanImages.push_back(img.image);
                                    }
                                    Log(fmt::format("[FrameInjection] Mapped {} injection images\n", count));
                                }
                            }
                        }
                    }

                    // Start RuntimeThread once holding pen + injection swapchain are both ready
                    if (!m_runtimeThread && m_holdingPen.IsInitialized() &&
                        m_frameInjection.IsReady() && !m_injectionVulkanImages.empty()) {
                        RuntimeThread::Config cfg{};
                        cfg.api                     = this;
                        cfg.session                 = m_session;
                        cfg.injectionSwapchain      = m_frameInjection.Swapchain();
                        cfg.injectionImages         = m_injectionVulkanImages;
                        cfg.device                  = m_vkDevice;
                        cfg.physDevice              = m_vkPhysicalDevice;
                        cfg.queue                   = m_vkQueue;
                        cfg.queueMutex              = &m_queueMutex;
                        cfg.queueFamilyIndex        = m_vkQueueFamilyIndex;
                        cfg.width                   = m_frameBroker.GetSwapchainWidth();
                        cfg.height                  = m_frameBroker.GetSwapchainHeight();
                        cfg.holdingPen              = &m_holdingPen;
                        cfg.xrWaitSwapchainImage    = m_xrWaitSwapchainImage;
                        cfg.xrReleaseSwapchainImage = m_xrReleaseSwapchainImage;
                        // Bind base-class methods directly so the runtime thread bypasses
                        // OpenXrLayer virtual overrides and reaches the real compositor.
                        cfg.waitFrame  = [this](XrSession s, const XrFrameWaitInfo* w, XrFrameState* f) {
                            return OpenXrApi::xrWaitFrame(s, w, f);
                        };
                        cfg.beginFrame = [this](XrSession s, const XrFrameBeginInfo* b) {
                            return OpenXrApi::xrBeginFrame(s, b);
                        };
                        cfg.endFrame   = [this](XrSession s, const XrFrameEndInfo* e) {
                            return OpenXrApi::xrEndFrame(s, e);
                        };

                        m_runtimeThread = std::make_unique<RuntimeThread>();
                        if (!m_runtimeThread->Start(std::move(cfg))) {
                            Log("[OpenXrLayer] Failed to start RuntimeThread\n");
                            m_runtimeThread.reset();
                        } else {
                            Log("[OpenXrLayer] RuntimeThread started\n");
                        }
                    }
                } else if (createInfo->usageFlags & XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                    m_frameBroker.RegisterSwapchain(*swapchain, *createInfo);
                    Log(fmt::format("Intercepted Depth Swapchain: {}x{}\n", createInfo->width, createInfo->height));
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
            if (session == m_session && m_runtimeThread && m_runtimeThread->IsRunning()) {
                // Runtime thread owns the real xrWaitFrame; give app thread a synthetic frame state.
                XrFrameState synth{XR_TYPE_FRAME_STATE};
                if (m_runtimeThread->GetSynthesizedFrameState(synth)) {
                    *frameState = synth;
                    m_poseProvider.OnWaitFrame(synth);
                    TraceLoggingWrite(g_traceProvider, "xrWaitFrame_Synthesized",
                                      TLArg(synth.predictedDisplayTime, "DisplayTime"));
                    return XR_SUCCESS;
                }
                // No data yet (first couple frames) — fall through to real call below.
            }
            XrResult result = OpenXrApi::xrWaitFrame(session, frameWaitInfo, frameState);
            if (XR_SUCCEEDED(result) && session == m_session && frameState) {
                m_poseProvider.OnWaitFrame(*frameState);
            }
            return result;
        }

        // 4. HOOK END FRAME
        XrResult xrEndFrame(XrSession session, const XrFrameEndInfo* frameEndInfo) override {
            if (session == m_session && m_apiType == GraphicsAPI::Vulkan) {
                // Phase 3: capture pose/FOV context for future phases only.
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

                            }
                            break; // Only process first projection layer
                        }
                    }
                }

                // Phase 3: pass-through only. No synthesis, no rewrite, no CUDA.
                TraceLoggingWrite(g_traceProvider, "xrEndFrame_PassThrough",
                                  TLArg(frameEndInfo ? frameEndInfo->layerCount : 0u, "LayerCount"));

                // Phase 3 Step A: copy current color image into holding pen slot.
                // Pass-through to OpenXR still happens below — runtime thread not yet running.
                if (m_holdingPen.IsInitialized() && projectionLayer != nullptr) {
                    VkImage currentColor = m_frameBroker.GetCurrentColorImage();
                    if (currentColor != VK_NULL_HANDLE) {
                        m_holdingPen.PollSlots();
                        const int32_t slot = m_holdingPen.AcquireIdleSlot();
                        if (slot >= 0) {
                            const XrTime displayTime = m_poseProvider.GetLastPredictedDisplayTime();
                            const bool copied = m_holdingPen.SubmitCopy(
                                slot, currentColor,
                                m_frameBroker.GetSwapchainWidth(), m_frameBroker.GetSwapchainHeight(),
                                displayTime, m_vkQueue, m_queueMutex);
                            TraceLoggingWrite(g_traceProvider, "HoldingPen_Copy",
                                              TLArg(slot, "Slot"),
                                              TLArg(copied, "Submitted"),
                                              TLArg(m_holdingPen.GetDebugState().c_str(), "State"));
                        } else {
                            TraceLoggingWrite(g_traceProvider, "HoldingPen_NoSlot",
                                              TLArg(m_holdingPen.GetDebugState().c_str(), "State"));
                        }
                    }
                }
            } // end if (session == m_session && Vulkan)

            // When runtime thread is running, app's xrEndFrame stops here.
            if (m_runtimeThread && m_runtimeThread->IsRunning()) {
                TraceLoggingWrite(g_traceProvider, "xrEndFrame_AppIntercepted",
                                  TLArg(true, "RuntimeThreadOwns"));
                return XR_SUCCESS;  // Do NOT forward — runtime thread owns compositor submission.
            }
            return OpenXrApi::xrEndFrame(session, frameEndInfo);
        }

        XrResult xrBeginFrame(XrSession session, const XrFrameBeginInfo* frameBeginInfo) override {
            if (session == m_session && m_runtimeThread && m_runtimeThread->IsRunning()) {
                // Runtime thread owns xrBeginFrame; app thread call is a no-op.
                return XR_SUCCESS;
            }
            return OpenXrApi::xrBeginFrame(session, frameBeginInfo);
        }

        XrResult xrDestroySession(XrSession session) override {
            if (session == m_session) {
                if (m_runtimeThread) {
                    Log("[OpenXrLayer] Stopping RuntimeThread on session destroy\n");
                    m_runtimeThread->Stop();
                    m_runtimeThread.reset();
                }
                m_holdingPen.Shutdown();
                m_session = XR_NULL_HANDLE;
            }
            return OpenXrApi::xrDestroySession(session);
        }

      private:
        void ResolveVulkanExtensionFunctions(XrInstance instance) {
            if (instance == XR_NULL_HANDLE || m_xrGetInstanceProcAddr == nullptr) {
                return;
            }

            PFN_xrVoidFunction function = nullptr;
            if (m_xrGetVulkanInstanceExtensionsKHR == nullptr &&
                XR_SUCCEEDED(m_xrGetInstanceProcAddr(instance, "xrGetVulkanInstanceExtensionsKHR", &function))) {
                m_xrGetVulkanInstanceExtensionsKHR = reinterpret_cast<PFN_xrGetVulkanInstanceExtensionsKHR>(function);
            }

            function = nullptr;
            if (m_xrGetVulkanDeviceExtensionsKHR == nullptr &&
                XR_SUCCEEDED(m_xrGetInstanceProcAddr(instance, "xrGetVulkanDeviceExtensionsKHR", &function))) {
                m_xrGetVulkanDeviceExtensionsKHR = reinterpret_cast<PFN_xrGetVulkanDeviceExtensionsKHR>(function);
            }
        }

        bool isSystemHandled(XrSystemId systemId) const { return systemId == m_systemId; }

        enum class GraphicsAPI { Unknown, DX11, Vulkan };
        GraphicsAPI m_apiType{GraphicsAPI::Unknown};
        bool m_bypassApiLayer{false};
        XrSystemId m_systemId{XR_NULL_SYSTEM_ID};
        XrSession m_session{XR_NULL_HANDLE};

        PFN_xrGetVulkanInstanceExtensionsKHR m_xrGetVulkanInstanceExtensionsKHR{nullptr};
        PFN_xrGetVulkanDeviceExtensionsKHR m_xrGetVulkanDeviceExtensionsKHR{nullptr};

        FrameBroker m_frameBroker;
        PoseProvider m_poseProvider;
        DepthProvider m_depthProvider;
        FrameContext m_frameContext{};

        PFN_xrWaitSwapchainImage m_xrWaitSwapchainImage{nullptr};
        PFN_xrReleaseSwapchainImage m_xrReleaseSwapchainImage{nullptr};

        // Vulkan handles (captured from xrCreateSession)
        VkInstance m_vkInstance{VK_NULL_HANDLE};
        VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};
        VkDevice m_vkDevice{VK_NULL_HANDLE};
        uint32_t m_vkQueueFamilyIndex{0};
        uint32_t m_vkQueueIndex{0};

        // Phase 3: holding pen + queue ownership
        HoldingPen m_holdingPen;
        std::mutex m_queueMutex;   // Serializes vkQueueSubmit between app and runtime threads
        VkQueue    m_vkQueue{VK_NULL_HANDLE};

        // Phase 3: runtime thread + injection swapchain
        std::unique_ptr<RuntimeThread> m_runtimeThread;
        FrameInjection                 m_frameInjection;
        std::vector<VkImage>           m_injectionVulkanImages;

        // Per-eye FOV data (extracted from xrEndFrame projection layers)
        XrFovf m_fovLeft{};
        XrFovf m_fovRight{};
        bool m_fovInitialized{false};
    };

    XrResult XRAPI_CALL xrGetVulkanInstanceExtensionsKHR_intercept(XrInstance instance,
                                                                    XrSystemId systemId,
                                                                    uint32_t bufferCapacityInput,
                                                                    uint32_t* bufferCountOutput,
                                                                    char* buffer) {
        auto* layer = dynamic_cast<OpenXrLayer*>(GetInstance());
        if (layer == nullptr) {
            return XR_ERROR_RUNTIME_FAILURE;
        }
        return layer->xrGetVulkanInstanceExtensionsKHR(instance, systemId, bufferCapacityInput, bufferCountOutput, buffer);
    }

    XrResult XRAPI_CALL xrGetVulkanDeviceExtensionsKHR_intercept(XrInstance instance,
                                                                  XrSystemId systemId,
                                                                  uint32_t bufferCapacityInput,
                                                                  uint32_t* bufferCountOutput,
                                                                  char* buffer) {
        auto* layer = dynamic_cast<OpenXrLayer*>(GetInstance());
        if (layer == nullptr) {
            return XR_ERROR_RUNTIME_FAILURE;
        }
        return layer->xrGetVulkanDeviceExtensionsKHR(instance, systemId, bufferCapacityInput, bufferCountOutput, buffer);
    }

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
