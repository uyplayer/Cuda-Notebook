//
// Created by uyplayer on 2024-06-17.
//


#include <error.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>


__global__ void clock_block(clock_t *d_o, clock_t clock_count) {
    unsigned int start_clock = (unsigned int) clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count) {
        unsigned int end_clock = (unsigned int) clock();
        clock_offset = (clock_t) (end_clock - start_clock);
    }
    d_o[0] = clock_offset;
}

__global__ void sum(clock_t *d_clocks, int N) {
    // 获取当前线程块（CTA）的句柄
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();
    __shared__ clock_t s_clocks[32];
    clock_t my_sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        my_sum += d_clocks[i];
    }

    s_clocks[threadIdx.x] = my_sum;
    cooperative_groups::sync(cta);
    for (int i = 16; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
        }
        // 同步线程块中的所有线程，确保每个线程完成了上一步的加法操作
        cooperative_groups::sync(cta);
    }

    d_clocks[0] = s_clocks[0];
}

int main() {
    std::cout << "Hello, concurrentKernels !" << std::endl;


    int nkernels = 8; // number of concurrent kernels
    int nstreams = nkernels + 1; // use one more stream than concurrent kernel
    int nbytes = nkernels * sizeof(clock_t); // number of data bytes
    float kernel_time = 10; // time the kernel should run in ms
    float elapsed_time; // timing variables
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

    cudaEvent_t *kernelEvent = new cudaEvent_t[nkernels];
    for (int i = 0; i < nkernels; i++) {
        HANDLE_ERROR(cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
    }
    clock_t total_clocks = 0;
#if defined(__arm__) || defined(__aarch64__)
    clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 100));
#else
    clock_t time_clocks = (clock_t) (kernel_time * deviceProp.clockRate);
#endif
    // 任何在默认流上执行的操作，都会在开始时等待默认流上的所有先前操作完成，并且所有其他流上的操作完成。
    cudaEventRecord(start_event, 0);
    for (int i = 0; i < nkernels; ++i) {
        clock_block<<<1, 1, 0, streams[i]>>>(&d_a[i], time_clocks);
        total_clocks += time_clocks;
        HANDLE_ERROR(cudaEventRecord(kernelEvent[i], streams[i]));
        // 等待其他事件
        HANDLE_ERROR(cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[i], 0));
    }

    sum<<<1, 32, 0, streams[nstreams - 1]>>>(d_a, nkernels);
    HANDLE_ERROR(cudaMemcpyAsync(a, d_a, sizeof(clock_t), cudaMemcpyDeviceToHost, streams[nstreams - 1]));

    HANDLE_ERROR(cudaEventRecord(stop_event, 0));
    // 等待事件
    HANDLE_ERROR(cudaEventSynchronize(stop_event));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

    std::cout << "Expected time for serial execution of " << nkernels << " kernels = " << nkernels * kernel_time /
            1000.0f << "s\n";
    std::cout << "Expected time for concurrent execution of " << nkernels << " kernels = " << kernel_time / 1000.0f <<
            "s\n";
    std::cout << "Measured time for sample = " << elapsed_time / 1000.0f << "s\n";


    for (int i = 0; i < nkernels; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(kernelEvent[i]);
    }
    delete streams;
    delete kernelEvent;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFreeHost(a);
    cudaFree(d_a);
    exit(EXIT_SUCCESS);
}
