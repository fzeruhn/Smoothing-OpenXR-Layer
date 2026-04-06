#pragma once

#include <cuda.h>
#include <cstdint>

extern "C" bool launch_rgba_to_gray(CUarray srcRgba,
                                     CUdeviceptr dstGray,
                                     uint32_t width,
                                     uint32_t height,
                                     size_t dstPitch,
                                     CUstream stream);

extern "C" bool launch_copy_rgba_array(CUarray srcRgba,
                                         CUarray dstRgba,
                                         uint32_t width,
                                         uint32_t height,
                                         CUstream stream);
