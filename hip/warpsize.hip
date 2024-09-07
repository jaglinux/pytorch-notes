#include <hip/hip_runtime.h>
int main() {
  printf("warpSize: %d, __AMDGCN_WAVEFRONT_SIZE: %d\n", warpSize, __AMDGCN_WAVEFRONT_SIZE);
}
