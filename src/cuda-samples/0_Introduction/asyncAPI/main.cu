//
// Created by uyplayer on 2024-06-13.
//
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cxxopts.hpp"
#include <error.h>

__global__ void increment_kernel(int *g_data, int inc_value,int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        g_data[idx] = g_data[idx] + inc_value;
    }
}


int main(int argc, char **argv) {
    std::cout << "Hello, asyncAPI!" << std::endl;

    // 获取设备ID
    cxxopts::Options options("asyncAPI", "A brief description");
    options.add_options()
        ("d,device", "Device ID", cxxopts::value<int>()->default_value("0"));
    const auto command_result = options.parse(argc, argv);
    int device_id = command_result["device"].as<int>();
    std::cout << "Device ID: " << device_id << std::endl;

    constexpr int n = 16 * 1024 * 1024;
    constexpr int nbytes = n * sizeof(int);
    int *h_a, *d_b;

    HANDLE_ERROR(cudaMallocHost(&h_a, nbytes));
    HANDLE_ERROR(cudaMemset(h_a, 0, nbytes));
    HANDLE_ERROR(cudaMalloc(&d_b, nbytes));
    HANDLE_ERROR(cudaMemset(d_b, 255, nbytes));

    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3((n + threads.x - 1) / threads.x, 1);

    cudaStream_t stream;
    HANDLE_ERROR(cudaStreamCreate(&stream));

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    float gpu_time = 0.0f;

    HANDLE_ERROR(cudaProfilerStart());
    HANDLE_ERROR(cudaEventRecord(start, stream));
    HANDLE_ERROR(cudaMemcpyAsync(d_b, h_a, nbytes, cudaMemcpyHostToDevice, stream));
    increment_kernel<<<blocks, threads, 0, stream>>>(d_b, 963, n);
    HANDLE_ERROR(cudaMemcpyAsync(h_a, d_b, nbytes, cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaEventRecord(stop, stream));
    HANDLE_ERROR(cudaProfilerStop());
    // HANDLE_ERROR(cudaStreamSynchronize(stream));
    unsigned long int counter = 0;
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        std::cout << "Waiting for GPU to finish... " << counter << std::endl;
        counter++;
    }
    HANDLE_ERROR(cudaEventElapsedTime(&gpu_time, start, stop));
    std::cout << "Time spent on GPU: " << gpu_time << " ms" << std::endl;

    HANDLE_ERROR(cudaFreeHost(h_a));
    HANDLE_ERROR(cudaFree(d_b));
    HANDLE_ERROR(cudaStreamDestroy(stream));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    std::cout << "Completed successfully!" << std::endl;
    return 0;
}