#include "gpu_pipeline_kernels.h"

#include <cuda_runtime.h>

__global__ void kernel_rgba_to_gray(cudaSurfaceObject_t src, uint8_t* dst, size_t dstPitch, uint32_t width, uint32_t height) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar4 rgba{};
    surf2Dread(&rgba, src, static_cast<int>(x * sizeof(uchar4)), static_cast<int>(y));
    const float gray = 0.299f * static_cast<float>(rgba.x) +
                       0.587f * static_cast<float>(rgba.y) +
                       0.114f * static_cast<float>(rgba.z);
    uint8_t* row = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(dst) + y * dstPitch);
    row[x] = static_cast<uint8_t>(gray + 0.5f);
}

extern "C" bool launch_rgba_to_gray(CUarray srcRgba,
                                     CUdeviceptr dstGray,
                                     uint32_t width,
                                     uint32_t height,
                                     size_t dstPitch,
                                     CUstream stream) {
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = reinterpret_cast<cudaArray_t>(srcRgba);

    cudaSurfaceObject_t srcSurf = 0;
    if (cudaCreateSurfaceObject(&srcSurf, &rd) != cudaSuccess) {
        return false;
    }

    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    kernel_rgba_to_gray<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        srcSurf,
        reinterpret_cast<uint8_t*>(dstGray),
        dstPitch,
        width,
        height);

    const cudaError_t launchResult = cudaGetLastError();
    cudaDestroySurfaceObject(srcSurf);
    return launchResult == cudaSuccess;
}

extern "C" bool launch_copy_rgba_array(CUarray srcRgba,
                                         CUarray dstRgba,
                                         uint32_t width,
                                         uint32_t height,
                                         CUstream stream) {
    cudaMemcpy3DParms copy{};
    copy.srcArray = reinterpret_cast<cudaArray_t>(srcRgba);
    copy.dstArray = reinterpret_cast<cudaArray_t>(dstRgba);
    copy.extent = make_cudaExtent(width, height, 1);
    copy.kind = cudaMemcpyDeviceToDevice;
    const cudaError_t result = cudaMemcpy3DAsync(&copy, reinterpret_cast<cudaStream_t>(stream));
    return result == cudaSuccess;
}
