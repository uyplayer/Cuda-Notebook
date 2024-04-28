/*
 * @Author: uyplayer
 * @Date: 2024/4/28 19:29
 * @Email: uyplayer@qq.com
 * @File: static.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/CUDA-Programming/memory
 * @Project_Name: Cuda-Notebook
 * @Description:
 */


#include <stdio.h>
#include <iostream>
#include <error.h>
#include <GpuTimer.h>

__device__ int d_x = 0;
__device__ int d_y[2];


void __global__ my_kernel() {
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);

}


void main_static() {
    int h_y[2] = {10, 20};
    // give value to d_x
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));

    my_kernel<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);

}