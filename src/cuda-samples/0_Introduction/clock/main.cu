//
// Created by uyplayer on 2024-06-16.
//

#include <iostream>
#include <cuda_runtime.h>
#include "error.h"
#include "cxxopts.hpp"


#define NUM_BLOCKS 64
#define NUM_THREADS 256


__global__ static void timedReduction(const float *input, float *output, clock_t *timer) {
    // 使用 extern __shared__ 声明共享内存
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // 在第一个线程记录开始时间
    if (tid == 0) timer[bid] = clock();

    // 复制输入到共享内存
    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    // 执行归约操作，找到最小值
    for (int d = blockDim.x; d > 0; d /= 2) {
        __syncthreads();

        if (tid < d) {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0) {
                shared[tid] = f1;
            }
        }
    }

    // 在第一个线程写回结果
    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    // 在第一个线程记录结束时间
    if (tid == 0) timer[bid + gridDim.x] = clock();
}


int main(int argc, char **argv) {
    printf("CUDA Clock sample\n");


    float *dinput = NULL;
    float *doutput = NULL;
    clock_t *dtimer = NULL;

    clock_t timer[NUM_BLOCKS * 2];
    float input[NUM_THREADS * 2];

    // 初始化输入数据
    for (int i = 0; i < NUM_THREADS * 2; i++) {
        input[i] = (float) i;
    }
    // 分配设备内存
    HANDLE_ERROR(cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2));
    HANDLE_ERROR(cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS));
    HANDLE_ERROR(cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
    // 将输入数据复制到设备
    HANDLE_ERROR(cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2, cudaMemcpyHostToDevice));
    // 启动核函数
    timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(dinput, doutput, dtimer);
    // 将计时器数据从设备复制到主机
    HANDLE_ERROR(cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost));
    // 释放设备内存
    HANDLE_ERROR(cudaFree(dinput));
    HANDLE_ERROR(cudaFree(doutput));
    HANDLE_ERROR(cudaFree(dtimer));
    // 计算平均时钟周期
    long double avgElapsedClocks = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        avgElapsedClocks += (long double) (timer[i + NUM_BLOCKS] - timer[i]);
    }
    avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avgElapsedClocks);

    return EXIT_SUCCESS;
}
