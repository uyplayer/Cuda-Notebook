#include <iostream>
#include "error.h"
#include "GpuTimer.h"
#include <cuda_runtime.h>

__global__ void add(float *a, float *b, float *c, int N) {
    int tid = blockIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void init(float *a, float *b, int N) {
    int tid = blockIdx.x;
    if (tid < N)
    {
        a[tid] = 1.0 * tid;
        b[tid] = (1.0 * tid * tid);
    }
}

void add_vector() {
    std::cout << "Hello, add_vector !" << std::endl;

    constexpr int N = 1 << 20;
    constexpr size_t bytes = N * sizeof(float);

    float *h_c;
    HANDLE_ERROR(cudaMallocHost(&h_c, bytes))


    float *d_a, *d_b, *d_c;
    HANDLE_ERROR(cudaMalloc(&d_a, bytes));
    HANDLE_ERROR(cudaMalloc(&d_b, bytes));
    HANDLE_ERROR(cudaMalloc(&d_c, bytes));

    init<<<N, 1>>>(d_a, d_b, N);
    add<<<N, 1>>>(d_a, d_b, d_c, N);

    HANDLE_ERROR(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));






    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_c);
}

