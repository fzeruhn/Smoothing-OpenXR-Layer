#include <cuda_runtime.h>
#include <cuda.h>

// Writes a deterministic pattern to a 2D RGBA surface.
// Each pixel's R channel = (x + y * width) % 256. G, B = 0, A = 255.
// This pattern is verifiable on CPU without floating-point comparison.
static __global__ void fill_pattern(cudaSurfaceObject_t surf,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uchar4 pixel = make_uchar4(
        static_cast<unsigned char>((x + y * width) % 256),
        0,
        0,
        255
    );
    surf2Dwrite(pixel, surf, x * sizeof(uchar4), y);
}

// C-callable launcher — main.cpp calls this since <<<>>> syntax is CUDA-only.
extern "C" void launch_fill_pattern(CUsurfObject surf, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    fill_pattern<<<grid, block>>>(static_cast<cudaSurfaceObject_t>(surf),
                                  width, height);
}
