

//
// Created by uyplayer on 2024/4/28.
//



#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cuda_runtime.h>
#include "error.h"
#include "GpuTimer.h"



const double c = 3.57;
const double EPSILON = 1.0e-15;


__global__ void  add(const double *x, const double *y, double *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}



__device__ double add1_device(const double x, const double y)
{
    return (x + y);
}


__global__ void add_if(const double *x, const double *y, double *z,const int N)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if( N <= index) return;

    z[index] = add1_device(x[index] , y[index]);
}




void check_1(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

void main_add1_cuda() {

    constexpr int N = 1 << 20;
    constexpr int M = sizeof(double) * N;


    double *h_X, *h_Y, *h_Z;
    h_X = (double *) malloc(M);
    h_Y = (double *) malloc(M);
    h_Z = (double *) malloc(M);



    for (int n = 0; n < N; ++n) {
        h_X[n] = rand() % 100;
        h_Y[n] = rand() % 100;
    }

    double *d_x, *d_y, *d_z;

    HANDLE_ERROR(cudaMalloc(&d_x, M));
    HANDLE_ERROR(cudaMalloc(&d_y, M));
    HANDLE_ERROR(cudaMalloc(&d_z, M));


    HANDLE_ERROR(cudaMemcpy(d_x, h_X, M, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_y, h_Y, M, cudaMemcpyHostToDevice));


    int block_size = 128;
    const int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    cudaMemcpy(h_Z, d_z, M, cudaMemcpyDeviceToHost);
    check_1(h_Z, N);

    cudaDeviceSynchronize();

    free(h_X);
    free(h_Y);
    free(h_Z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);


}

