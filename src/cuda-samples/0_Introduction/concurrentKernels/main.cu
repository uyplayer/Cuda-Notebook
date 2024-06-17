//
// Created by uyplayer on 2024-06-17.
//



#include <error.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>


int main() {
    std::cout << "Hello, concurrentKernels !" << std::endl;


    int nkernels = 8;             // number of concurrent kernels
    int nstreams = nkernels + 1;  // use one more stream than concurrent kernel
    int nbytes = nkernels * sizeof(clock_t);  // number of data bytes
    float kernel_time = 10;                   // time the kernel should run in ms
    float elapsed_time;                       // timing variables
    int cuda_device = 0;


    HANDLE_ERROR(cudaGetDevice(&cuda_device));
    cudaDeviceProp deviceProp;
    HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, cuda_device));

    if (!deviceProp.cooperativeLaunch) {
        std::cerr << "Error: device does not support cooperative kernel launch" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (deviceProp.concurrentKernels == 0) {
        std::cerr << "Error: device does not support concurrent kernel execution" << std::endl;
        exit(EXIT_FAILURE);
    }


    std::cout << "CUDA device " << cuda_device << " : " << deviceProp.name << std::endl;
    std::cout << "Detected Compute SM " << deviceProp.major << "." << deviceProp.minor
              << " hardware with " << deviceProp.multiProcessorCount << " multi-processors";

    clock_t *a = nullptr;
    HANDLE_ERROR(cudaMallocHost((void **) &a, nbytes));

    clock_t *d_a = nullptr;
    HANDLE_ERROR(cudaMallocHost((void **) &d_a, nbytes));

    cudaStream_t *streams = new cudaStream_t[nstreams];
    for (int i = 0; i < nstreams; i++) {
        HANDLE_ERROR(cudaStreamCreate(&(streams[i])));
    }
    cudaEvent_t start_event, stop_event;
    HANDLE_ERROR(cudaEventCreate(&start_event));
    HANDLE_ERROR(cudaEventCreate(&stop_event));





    exit(EXIT_SUCCESS);
}