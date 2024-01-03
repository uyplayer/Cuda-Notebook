
#include "common.h"

// CUDA内核函数定义
__global__ void simpleKernel(int *data, int val) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] += val;
}

int cuda_stream_host_copy() {
    const int numElements = 1024;
    const int size = numElements * sizeof(int);
    const int numStreams = 5;
    int *d_data;

    // 分配设备内存
    cudaMalloc(&d_data, size);

    // 初始化设备内存
    cudaMemset(d_data, 0, size);

    // 创建5个CUDA流
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // 在每个流上执行内核
    for (int i = 0; i < numStreams; ++i) {
        simpleKernel<<<1, 1024, 0, streams[i]>>>(d_data, i);
    }

    // 同步所有流
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // 销毁所有流
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // 清理设备内存
    cudaFree(d_data);

    return 0;
}
