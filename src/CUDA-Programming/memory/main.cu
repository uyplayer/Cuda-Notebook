//
// Created by uyplayer on 2024/4/28.
//

#include <iostream>
#include <error.h>
#include <GpuTimer.h>

int main(int argc, char *argv[]){
    std::cout << "Hello memory" << std::endl;

    int device_id = 0;
    if (argc > 1) device_id = atoi(argv[1]);
    CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "Device id:                                 " << device_id << "\n";
    std::cout << "Device name:                               " << prop.name << "\n";
    std::cout << "Compute capability:                        " << prop.major << "." << prop.minor << "\n";
    std::cout << "Amount of global memory:                   " << prop.totalGlobalMem / (1024.0 * 1024 * 1024) << " GB\n";
    std::cout << "Amount of constant memory:                 " << prop.totalConstMem  / 1024.0 << " KB\n";
    std::cout << "Maximum grid size:                         " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << "\n";
    std::cout << "Maximum block size:                        " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << "\n";
    std::cout << "Number of SMs:                             " << prop.multiProcessorCount << "\n";
    std::cout << "Maximum amount of shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
    std::cout << "Maximum amount of shared memory per SM:    " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB\n";
    std::cout << "Maximum number of registers per block:     " << prop.regsPerBlock / 1024 << " K\n";
    std::cout << "Maximum number of registers per SM:        " << prop.regsPerMultiprocessor / 1024 << " K\n";
    std::cout << "Maximum number of threads per block:       " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Maximum number of threads per SM:          " << prop.maxThreadsPerMultiProcessor << "\n";

    return 0;
}