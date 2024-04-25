/*
 * @Author: uyplayer
 * @Date: 2024/4/25 21:08
 * @Email: uyplayer@qq.com
 * @File: vectorAdd_um_baseline.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/vector_addition/unified_memory
 * @Project_Name: Cuda-Notebook
 * @Description:
 */


#include "error.h"
#include "GpuTimer.h"
#include <iostream>
#include <assert.h>


__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    // Calculate global thread thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;


    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    std::cout << "Hello, vectorAdd_um_baseline !" << std::endl;


    constexpr int N = 1 << 20;   // 1M elements   size of 2^16
    constexpr size_t bytes = N * sizeof(float);

    // Declare unified memory pointers
    int *d_a, *d_b, *d_c;

    // Allocate unified memory for these pointers
    // when using cudaMallocManaged, we don't need to copy data between host and device
    HANDLE_ERROR(cudaMallocManaged(&d_a, bytes));
    HANDLE_ERROR(cudaMallocManaged(&d_b, bytes));
    HANDLE_ERROR(cudaMallocManaged(&d_c, bytes));



    // Initialize vectors
    // when writing to a unified memory pointer,
    // the data is immediately available to both host and device
    for (int i = 0; i < N; i++) {
        d_a[i] = rand() % 100;
        d_b[i] = rand() % 100;
    }

    int BLOCK_SIZE = 1 << 10;
    int GRID_SIZE = (N + BLOCK_SIZE - 1);

    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);

    // blocks the CPU until all previously issued commands in the CUDA stream have completed
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        assert(d_c[i] == d_a[i] + d_b[i]);
    }

    // Free unified memory (same as memory allocated with cudaMalloc)
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_c);


    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;


    return 0;


}