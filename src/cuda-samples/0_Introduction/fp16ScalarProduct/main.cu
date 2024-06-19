//
// Created by uyplayer on 2024-06-19.
//




#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "error.h"


#define NUM_OF_BLOCKS 128
#define NUM_OF_THREADS 256


void generateInput(half2 *a, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        half2 temp;
        temp.x = static_cast<float>(rand() % 4);
        temp.y = static_cast<float>(rand() % 2);
        a[i] = temp;
    }
}


int main() {
    std::cout << "Hello, fp16ScalarProduct !" << std::endl;

    // 使用BF16 数据类型
    size_t size = NUM_OF_BLOCKS * NUM_OF_THREADS * 16;
    half2 *vec[2];
    half2 *devVec[2];

    float *results;
    float *devResults;

    int device = 0;
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
    HANDLE_ERROR(cudaSetDevice(device));

    for (int i = 0; i < 2; ++i) {
        HANDLE_ERROR(cudaMallocHost((void **) &vec[i], size * sizeof *vec[i]));
        HANDLE_ERROR(cudaMalloc((void **) &devVec[i], size * sizeof *devVec[i]));

    }

    HANDLE_ERROR(cudaMallocHost((void **) &results, NUM_OF_BLOCKS * sizeof *results));
    HANDLE_ERROR(cudaMalloc((void **) &devResults, NUM_OF_BLOCKS * sizeof *devResults));

    for (int i = 0; i < 2; ++i) {
        generateInput(vec[i], size);
        HANDLE_ERROR(cudaMemcpy(devVec[i], vec[i], size * sizeof *vec[i],cudaMemcpyHostToDevice));
    }
    return 0;
}