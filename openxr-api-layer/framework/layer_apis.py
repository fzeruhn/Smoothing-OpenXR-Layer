# The list of OpenXR functions our layer will override.
override_functions = [
    "xrGetSystem",
    "xrCreateSession",
    "xrCreateVulkanDeviceKHR",
    "xrDestroySession",
    "xrCreateSwapchain",
    "xrEnumerateSwapchainImages",
    "xrWaitFrame",
    "xrEndFrame",
    "xrAcquireSwapchainImage",
    "xrWaitSwapchainImage",
    "xrReleaseSwapchainImage",
    "xrEndSession",
    "xrLocateViews",
    "xrSyncActions",
]

# The list of OpenXR functions our layer will use from the runtime.
# Might repeat entries from override_functions above.
requested_functions = [
    "xrGetInstanceProperties",
    "xrGetSystemProperties",
    "xrLocateViews",
    "xrSyncActions",
]

# The list of OpenXR extensions our layer will either override or use.
extensions = [
    "XR_KHR_vulkan_enable2",
]
