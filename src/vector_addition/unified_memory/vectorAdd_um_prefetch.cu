/*
 * @Author: uyplayer
 * @Date: 2024/4/25 20:04
 * @Email: uyplayer@qq.com
 * @File: vectorAdd_um.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/vectorAdd_um
 * @Project_Name: Cuda-Notebook
 * @Description:
 */

// https://giahuy04.medium.com/unified-memory-81bb7c0f0270


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
    std::cout << "Hello, vectorAdd_um !" << std::endl;


    constexpr int N = 1 << 20;   // 1M elements   size of 2^16
    constexpr size_t bytes = N * sizeof(float);


    // Declare unified memory pointers
    int *d_a, *d_b, *d_c;


    // Allocate unified memory for these pointers
    // when using cudaMallocManaged, we don't need to copy data between host and devicece
    HANDLE_ERROR(cudaMallocManaged(&d_a, bytes));
    HANDLE_ERROR(cudaMallocManaged(&d_b, bytes));
    HANDLE_ERROR(cudaMallocManaged(&d_c, bytes));


    // get device id
    int id = cudaGetDevice(&id);

    // 这两行代码使用cudaMemAdvise函数来设置内存的首选位置。这里，它们将输入向量a和b的首选位置设置为CPU
    cudaMemAdvise(d_a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(d_b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    // 这两行代码使用cudaMemPrefetchAsync函数将数据异步地从CPU复制到GPU
    // 这行代码使用cudaMemPrefetchAsync函数来预取内存。这里，它预取输出向量c到当前活动的设备（GPU）。
    // 预取操作是异步的，它不会阻塞CPU，但会在必要时将数据从CPU迁移到GPU
    cudaMemPrefetchAsync(d_c, bytes, id);




    // Initialize vectors
    // when writing to a unified memory pointer,
    // the data is immediately available to both host and device
    for (int i = 0; i < N; i++) {
        d_a[i] = rand() % 100;
        d_b[i] = rand() % 100;
    }

    // 这两行代码使用cudaMemAdvise函数来设置内存的首选位置。这里，它们将输入向量a和b的首选位置设置为GPU
    cudaMemAdvise(d_a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(d_b, bytes, cudaMemAdviseSetReadMostly, id);

    // 这两行代码使用cudaMemPrefetchAsync函数来预取内存。这里，它预取输入向量a和b到当前活动的设备（GPU）。
    // 预取操作是异步的，它不会阻塞CPU，但会在必要时将数据从CPU迁移到GPU。
    cudaMemPrefetchAsync(d_a, bytes, id);
    cudaMemPrefetchAsync(d_b, bytes, id);



    int BLOCK_SIZE = 1 << 10;
    int GRID_SIZE = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;

    // Launch the kernel
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();


    // Prefetch to the host (CPU)
    cudaMemPrefetchAsync(d_a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(d_b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(d_c, bytes, cudaCpuDeviceId);

    // Verify the result on the CPU
    for (int i = 0; i < N; i++) {
        assert(d_c[i] == d_a[i] + d_b[i]);
    }

    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;
    return 0;
}