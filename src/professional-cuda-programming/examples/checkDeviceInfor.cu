

//
// Created by uyplayer on 2024/4/29.
//




#include <iostream>
#include <cuda_runtime.h>
#include "error.h"


void checkDeviceInfor() {
    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA" << std::endl;
        exit(0);
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp{};
        CHECK(cudaGetDeviceProperties(&deviceProp, i));

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Total amount of global memory: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "  Total amount of constant memory: " << deviceProp.totalConstMem << std::endl;
        std::cout << "  Total amount of shared memory per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "  Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum sizes of each dimension of a block: " << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "  Maximum sizes of each dimension of a grid: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "  Maximum memory pitch: " << deviceProp.memPitch << std::endl;
        std::cout << "  Memory Clock Rate (KHz): " << deviceProp.memoryClockRate << std::endl;
        std::cout << "  Memory Bus Width (bits): " << deviceProp.memoryBusWidth << std::endl;
        std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6 << std::endl;
    }
}