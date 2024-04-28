

//
// Created by uyplayer on 2024/4/28.
//


#include <iostream>
#include "error.h"



void __global__ add2gpu(const double *x, const double *y, double *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}



void add2gpu(){
    std::cout << "Hello, add2gpu!" << std::endl;

    constexpr int N = 1 << 20;
    size_t size = N * sizeof(double);

    double *h_x, *h_y, *h_z;


    h_x = new double[N];
    h_y = new double[N];
    h_z = new double[N];

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = rand() % 100;
        h_y[n] = rand() % 100;
    }

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    double *d_x, *d_y, *d_z;
    CHECK(cudaMalloc(&d_x, size));
    CHECK(cudaMalloc(&d_y, size));
    CHECK(cudaMalloc(&d_z, size));


    cudaEvent_t  start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    CHECK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));

    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    add2gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);


    CHECK(cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    std::cout << "h_z: " << h_z << std::endl;

    std::cout << "Elapsed time: " << elapsed_time << "ms" << std::endl;

    // free
    delete[]  h_x,h_y,h_z;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}