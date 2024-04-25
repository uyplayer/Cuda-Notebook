/*
 * @Author: uyplayer
 * @Date: 2024/4/25 19:34
 * @Email: uyplayer@qq.com
 * @File: vectorAdd.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/baseline
 * @Project_Name: Cuda-Notebook
 * @Description:
 */


#include "error.h"
#include "GpuTimer.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <assert.h>



__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,int *__restrict c, int N){

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) c[tid] = a[tid] + b[tid];

}


void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
    for (int i = 0; i < a.size(); i++) {
        assert(c[i] == a[i] + b[i]);
    }
}


int main() {

    std::cout << "Hello, baseline !" << std::endl;


    constexpr int N = 1 << 20;   // 1M elements   size of 2^16
    constexpr size_t bytes = N * sizeof(float);


    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> c(N);


    // initialize a and b arrays on the host
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // allocate the memory on the GPU

    int *d_a, *d_b, *d_c;

    HANDLE_ERROR(cudaMalloc(&d_a, bytes));
    HANDLE_ERROR(cudaMalloc(&d_b, bytes));
    HANDLE_ERROR(cudaMalloc(&d_c, bytes));

    // copy data from the host to the GPU
    HANDLE_ERROR(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_c, c.data(), bytes, cudaMemcpyHostToDevice));


    // kernel launch
    int numThreads = 1 << 10;
    int numBlocks = (N + numThreads - 1) / numThreads;

    // add the vectors on the GPU
    // Kernel calls are asynchronous operation
    vectorAdd<<<numBlocks, numThreads>>>(d_a, d_b, d_c, N);

    // copy the result back to the hosts
    // cudaMemcpy is a synchronous operation
    // cudaMemcpy acts as both a memcpy and synchronization barrier
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    // verify the result
    verify_result(a, b, c);

    // free the memory allocated on the GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

    return 0;
}