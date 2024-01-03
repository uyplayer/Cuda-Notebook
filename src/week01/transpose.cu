
#include "common.h"

// 设备函数，用于交换两个元素
__device__ void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

// 核函数，用于执行矩阵转置
__global__ void transpose(float* out, const float* in, const int rows, const int cols) {
    auto xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    auto yIndex = blockIdx.y * blockDim.y + threadIdx.y;


    auto index_in = xIndex + rows * yIndex;
    auto index_out = yIndex + cols * xIndex;

    // 检查边界条件
    if (xIndex < cols && yIndex < rows) {
        out[index_out] = in[index_in];  // 交换操作
    }
}

int main() {
    // 定义矩阵的行数和列数
    const int rows = 1024;
    const int cols = 1024;

    // 分配和初始化矩阵
    auto *h_in = (float*)malloc(rows * cols * sizeof(float));
    auto *h_out = (float*)malloc(rows * cols * sizeof(float));

    // 初始化矩阵 h_in
    for(int i = 0; i < rows * cols; i++) {
        h_in[i] = float(i);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_in, h_in, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块大小和网格大小
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // 启动核函数
    transpose<<<grid, block>>>(d_out, d_in, rows, cols);

    // 等待GPU完成工作
    cudaDeviceSynchronize();

    // 拷贝回结果
    cudaMemcpy(h_out, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
