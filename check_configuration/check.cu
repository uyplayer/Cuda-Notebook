

#include "../common/common.h"

int main() {
    int driver_version = 0, runtime_version = 0;
    cudaError_t status;

    // 获取CUDA驱动版本
    status = cudaDriverGetVersion(&driver_version);
    if (status != cudaSuccess) {
        std::cerr << "Failed to get CUDA driver version: " << cudaGetErrorString(status) << std::endl;
        return 1;
    }

    // 获取CUDA运行时版本
    status = cudaRuntimeGetVersion(&runtime_version);
    if (status != cudaSuccess) {
        std::cerr << "Failed to get CUDA runtime version: " << cudaGetErrorString(status) << std::endl;
        return 1;
    }

    std::cout << "CUDA Driver Version: " << driver_version << std::endl;
    std::cout << "CUDA Runtime Version: " << runtime_version << std::endl;

    return 0;
}
