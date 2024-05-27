//
// Created by uyplayer on 2024-05-27.
//


#include <iostream>
#include <../common/error.h>
#include <cuda_runtime.h>


__global__ void v1(const float* x, float* y)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = x[tid];
    __syncthreads();

    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();


    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) sdata[tid] += sdata[tid + 32];
    __syncthreads();

    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncthreads();

    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncthreads();

    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncthreads();

    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncthreads();

    if (tid == 0)
    {
        *y = sdata[0] + sdata[1];
    }
}

void reduction_v1()
{
    std::cout << "Hallo Reduction v1" << std::endl;

    constexpr int n_size = 256;
    int byte_size = n_size * sizeof(float);

    float *d_x, *d_y, *h_x;
    HANDLE_ERROR(cudaMalloc(&d_x, byte_size));
    HANDLE_ERROR(cudaMalloc(&d_y, sizeof(float)));
    HANDLE_ERROR(cudaMallocHost(&h_x, byte_size));

    for (int i = 0; i < n_size; i++)
    {
        h_x[i] = 1.0f;
    }

    cudaMemcpy(d_x, h_x, byte_size, cudaMemcpyHostToDevice);
    v1<<<1, n_size>>>(d_x, d_y);
    cudaDeviceSynchronize();
    float result = 0;
    cudaMemcpy(&result, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(h_x);
}
