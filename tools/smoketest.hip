// A small HIP program to smoketest HIP runtime.
#include "hip/hip_runtime.h"
#include <iostream>

__global__ void smoketest(int *d) { *d = 123; }

#define CHECK_ERR                                                              \
  do {                                                                         \
    if (err == hipSuccess)                                                     \
      break;                                                                   \
    std::cerr << "Error at line " << __LINE__ << std::endl;                    \
    exit(1);                                                                   \
  } while (0)

int main() {

  hipDeviceProp_t devProps;
  hipError_t err = hipGetDeviceProperties(&devProps, 0);
  CHECK_ERR;

  std::cout << "Device: " << devProps.name << std::endl;

  int *devData;
  err = hipMalloc(&devData, sizeof(int));
  CHECK_ERR;

  int hostData = 0;
  err = hipMemcpy(devData, &hostData, sizeof(int), hipMemcpyHostToDevice);
  CHECK_ERR;

  smoketest<<<1, 1>>>(devData);
  if (hipGetLastError() != hipSuccess) {
    std::cerr << "Kernel launch failed!" << std::endl;
    return 1;
  }

  err = hipMemcpy(&hostData, devData, sizeof(int), hipMemcpyDeviceToHost);
  CHECK_ERR;

  if (hostData != 123) {
    std::cerr << "error: expected '" << 123 << "'. Got '" << hostData << "'"
              << std::endl;
    return 1;
  }

  return 0;
}
