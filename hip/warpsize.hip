#include <hip/hip_runtime.h>

__global__ void helloworld_kernel()
{
        printf("warpsize %d", __AMDGCN_WAVEFRONT_SIZE);
}


int main() {
    helloworld_kernel<<<dim3(1,1,1), // 3D grid specifying number of blocks to launch: (2, 1, 1)
                        dim3(1,1,1), // 3D grid specifying number of threads to launch: (2, 1, 1)
                        0, // number of bytes of additional shared memory to allocate
                        hipStreamDefault // stream where the kernel should execute: default stream
                        >>>();
   hipDeviceSynchronize();
}
