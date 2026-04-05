#include "pch.h"
#include "pose_provider.h"

namespace openxr_api_layer {

void PoseProvider::OnWaitFrame(const XrFrameState& frameState) {
    m_lastPredictedDisplayTime = frameState.predictedDisplayTime;
}

XrResult PoseProvider::PopulatePredictedViews(OpenXrApi& api,
                                              XrSession session,
                                              const XrCompositionLayerProjection& projectionLayer,
                                              FrameContext& frameContext) const {
    if (projectionLayer.viewCount < 2 || projectionLayer.space == XR_NULL_HANDLE) {
        return XR_ERROR_VALIDATION_FAILURE;
    }

    if (m_lastPredictedDisplayTime == 0) {
        return XR_ERROR_CALL_ORDER_INVALID;
    }

    frameContext.displayTime = m_lastPredictedDisplayTime;
    frameContext.projectionSpace = projectionLayer.space;

    XrViewLocateInfo locateInfo{XR_TYPE_VIEW_LOCATE_INFO};
    locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    locateInfo.displayTime = m_lastPredictedDisplayTime;
    locateInfo.space = projectionLayer.space;

    std::array<XrView, 2> locatedViews{
        XrView{XR_TYPE_VIEW},
        XrView{XR_TYPE_VIEW},
    };
    uint32_t viewCountOutput = 0;
    XrViewState viewState{XR_TYPE_VIEW_STATE};

    const XrResult locateResult =
        api.xrLocateViews(session, &locateInfo, &viewState, static_cast<uint32_t>(locatedViews.size()), &viewCountOutput, locatedViews.data());
    if (XR_FAILED(locateResult)) {
        return locateResult;
    }

    frameContext.viewStateFlags = viewState.viewStateFlags;

    for (uint32_t eye = 0; eye < 2; ++eye) {
        frameContext.predictedViews[eye].valid = eye < viewCountOutput;
        if (eye < viewCountOutput) {
            frameContext.predictedViews[eye].pose = locatedViews[eye].pose;
            frameContext.predictedViews[eye].fov = locatedViews[eye].fov;
        }
    }

    return XR_SUCCESS;
}

} // namespace openxr_api_layer
