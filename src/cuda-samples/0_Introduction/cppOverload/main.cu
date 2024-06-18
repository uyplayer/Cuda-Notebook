//
// Created by uyplayer on 2024-06-17.
//


#include <iostream>
#include <cuda_runtime.h>
#include "error.h"


#define THREAD_N 256
#define N 1024
#define DIV_UP(a,b) (((a) + (b) - 1) / (b))
#define OUTPUT_ATTR(attr)                                         \
    printf("Shared Size:   %d\n", (int)attr.sharedSizeBytes);       \
    printf("Constant Size: %d\n", (int)attr.constSizeBytes);        \
    printf("Local Size:    %d\n", (int)attr.localSizeBytes);        \
    printf("Max Threads Per Block: %d\n", attr.maxThreadsPerBlock); \
    printf("Number of Registers: %d\n", attr.numRegs);              \
    printf("PTX Version: %d\n", attr.ptxVersion);                   \
    printf("Binary Version: %d\n", attr.binaryVersion);

__global__ void simple_kernel(const int *pIn, int *pOut, int a) {
    __shared__ int sData[THREAD_N];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    sData[threadIdx.x] = pIn[tid];
    __syncthreads();
    pOut[tid] = sData[threadIdx.x] * a + tid;
}

__global__ void simple_kernel(const int2 *pIn, int *pOut, int a) {
    __shared__ int2 sData[THREAD_N];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    sData[threadIdx.x] = pIn[tid];
    __syncthreads();

    pOut[tid] = (sData[threadIdx.x].x + sData[threadIdx.x].y) * a + tid;;
}

__global__ void simple_kernel(const int *pIn1, const int *pIn2, int *pOut, int a) {
    __shared__ int sData1[THREAD_N];
    __shared__ int sData2[THREAD_N];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    sData1[threadIdx.x] = pIn1[tid];
    sData2[threadIdx.x] = pIn2[tid];
    __syncthreads();

    pOut[tid] = (sData1[threadIdx.x] + sData2[threadIdx.x]) * a + tid;
}

int main() {
    std::cout << "Hello, cppOverload!" << std::endl;

    int *hInput = nullptr;
    int *hOutput = nullptr;
    int *dInput = nullptr;
    int *dOutput = nullptr;

    int deviceCount;
    HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
    int device = 0;
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
    HANDLE_ERROR(cudaSetDevice(device));

    HANDLE_ERROR(cudaMalloc(&dInput, sizeof(int) * N * 2));
    HANDLE_ERROR(cudaMalloc(&dOutput, sizeof(int) * N));

    HANDLE_ERROR(cudaMallocHost(&hInput, sizeof(int) * N * 2));
    HANDLE_ERROR(cudaMallocHost(&hOutput, sizeof(int) * N));

    for (int i = 0; i < N * 2; ++i) {
        hInput[i] = i;
    }
    HANDLE_ERROR(cudaMemcpy(dInput, hInput, sizeof(int) * N * 2, cudaMemcpyHostToDevice));
    int a = 1;
    // overload function
    void (*func1)(const int *, int *, int);
    void (*func2)(const int2 *, int *, int);
    void (*func3)(const int *, const int *, int *, int);

    func1 = simple_kernel;
    // CUDA函数属性结构
    struct cudaFuncAttributes attr;
    // 指定函数指针并设置缓存配置
    HANDLE_ERROR(cudaFuncSetCacheConfig(*func1, cudaFuncCachePreferShared));
    // 获取函数属性
    HANDLE_ERROR(cudaFuncGetAttributes(&attr, *func1));
    OUTPUT_ATTR(attr);
    (*func1)<<<DIV_UP(N, THREAD_N), THREAD_N>>>(dInput, dOutput, a);
    HANDLE_ERROR(cudaMemcpy(hOutput, dOutput, sizeof(int) * N, cudaMemcpyDeviceToHost));
    // output hOutput
    for (int i = 0; i < N; ++i) {
        std::cout << hOutput[i] <<std::endl;
    }

    func2 = simple_kernel;
    HANDLE_ERROR(cudaFuncSetCacheConfig(*func2, cudaFuncCachePreferShared));
    HANDLE_ERROR(cudaFuncGetAttributes(&attr, *func2));
    OUTPUT_ATTR(attr);
    (*func2)<<<DIV_UP(N, THREAD_N), THREAD_N>>>((int2 *)dInput, dOutput, a);
    HANDLE_ERROR(cudaMemcpy(hOutput, dOutput, sizeof(int) * N, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        std::cout << hOutput[i] <<std::endl;
    }

    func3 = simple_kernel;
    memset(&attr, 0, sizeof(attr));
    HANDLE_ERROR(cudaFuncSetCacheConfig(*func3, cudaFuncCachePreferShared));
    HANDLE_ERROR(cudaFuncGetAttributes(&attr, *func3));
    OUTPUT_ATTR(attr);
    (*func3)<<<DIV_UP(N, THREAD_N), THREAD_N>>>(dInput, dInput + N, dOutput, a);
    HANDLE_ERROR(cudaMemcpy(hOutput, dOutput, sizeof(int) * N, cudaMemcpyDeviceToHost));
    cudaFree(dInput);
    cudaFree(dOutput);
    cudaFree(hInput);
    cudaFree(hOutput);
    return 0;
}
