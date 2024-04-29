

//
// Created by uyplayer on 2024/4/29.
//


#include <iostream>

// CUDA 核函数，将数组元素相加
__global__ void addKernel(int *a, int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int arraySize = 10000;
    const int arrayBytes = arraySize * sizeof(int);

    // 在主机内存中分配数组
    int *h_a = (int*)malloc(arrayBytes);
    int *h_b = (int*)malloc(arrayBytes);
    int *h_c = (int*)malloc(arrayBytes);

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // 在设备内存中分配数组
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, arrayBytes);
    cudaMalloc(&d_b, arrayBytes);
    cudaMalloc(&d_c, arrayBytes);

    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 将数据从主机内存传输到设备内存（异步）
    cudaMemcpyAsync(d_a, h_a, arrayBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, arrayBytes, cudaMemcpyHostToDevice, stream);

    // 调用 CUDA 核函数进行计算（异步）
    addKernel<<<(arraySize + 255) / 256, 256, 0, stream>>>(d_a, d_b, d_c, arraySize);

    // 将计算结果从设备内存传输到主机内存（异步）
    cudaMemcpyAsync(h_c, d_c, arrayBytes, cudaMemcpyDeviceToHost, stream);

    // 等待 CUDA 流中的所有操作完成
    cudaStreamSynchronize(stream);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // 释放资源
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaStreamDestroy(stream);

    return 0;
}
