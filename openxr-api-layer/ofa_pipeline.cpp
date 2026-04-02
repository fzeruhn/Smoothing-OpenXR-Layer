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
