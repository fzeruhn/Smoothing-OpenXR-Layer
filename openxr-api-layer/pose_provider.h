#pragma once

#include "frame_context.h"

namespace openxr_api_layer {

class OpenXrApi;

class PoseProvider {
  public:
    void OnWaitFrame(const XrFrameState& frameState);
    XrResult PopulatePredictedViews(OpenXrApi& api,
                                    XrSession session,
                                    const XrCompositionLayerProjection& projectionLayer,
                                    FrameContext& frameContext) const;

    XrTime GetLastPredictedDisplayTime() const { return m_lastPredictedDisplayTime; }

  private:
    XrTime m_lastPredictedDisplayTime{0};
};

} // namespace openxr_api_layer
