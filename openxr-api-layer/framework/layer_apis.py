# The list of OpenXR functions our layer will override.
override_functions = [
    "xrGetSystem",
    "xrCreateSession",
    "xrCreateSwapchain",
    "xrEnumerateSwapchainImages",
    "xrWaitFrame",
    "xrEndFrame",
    "xrAcquireSwapchainImage",
]

# The list of OpenXR functions our layer will use from the runtime.
# Might repeat entries from override_functions above.
requested_functions = [
    "xrGetInstanceProperties",
    "xrGetSystemProperties",
    "xrLocateViews",
]

# The list of OpenXR extensions our layer will either override or use.
extensions = []
