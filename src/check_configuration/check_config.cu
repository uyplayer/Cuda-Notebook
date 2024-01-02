

#include "common.h"

int main() {
    int dev;
    HANDLE_ERROR(cudaGetDevice(&dev));
    std::cout << "current device id : " << dev << std::endl;
    // choose device or set device
    cudaDeviceProp prop_choose{};
    prop_choose.major = 1;
    prop_choose.minor = 3;
    // set
    HANDLE_ERROR(cudaSetDevice(dev));
    // choose
    HANDLE_ERROR(cudaChooseDevice(&dev,&prop_choose));
    int count ;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    std::cout << "number of compute-capable devices : " << count << std::endl;

    cudaDeviceProp prop{};
    for (int i = 0;i<count;i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop,i));
        std::cout << "maxGridSize : " << prop.maxGridSize << std::endl;
        std::cout << "maxThreadsPerBlock : " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "maxBlocksPerMultiProcessor : " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "multiProcessorCount : " << prop.multiProcessorCount << std::endl;
    }

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

    printf("Hallow Cuda");
    return 0;
}
