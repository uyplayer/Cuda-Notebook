//
// Created by uyplayer on 2024-06-03.
//


#include <iostream>
#include <chrono>
#include "matmul.cuh"
#include "error.h"


__global__ void mat_mul(float *A, float *B, float *C, int row, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < row && idy < col) {
        float sum = 0.0f;
        for (int i = 0; i < col; i++) {
            sum += A[idx * col + i] * B[i + idy * col];
        }
        C[idx * col + idy] = sum;
    }
}


void matmul_gpu() {
    std::cout << "Matrix Multiplication on GPU" << std::endl;

    constexpr int Row = 5000;
    constexpr int Col = 50;

    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;

    h_A = new float[Row * Col];
    h_B = new float[Col * Row];
    h_C = new float[Row * Row];

    for (int i = 0; i < Row * Col; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    HANDLE_ERROR(cudaMalloc((void **) &d_A, Row * Col * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &d_B, Col * Row * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &d_C, Row * Row * sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(d_A, h_A, Row * Col * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, h_B, Col * Row * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((Row + threadsPerBlock.x - 1) / threadsPerBlock.x, (Row + threadsPerBlock.y - 1) / threadsPerBlock.y); // Using Row for the result matrix
    auto start = std::chrono::high_resolution_clock::now();
    mat_mul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, Row, Col);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s\n";

    HANDLE_ERROR(cudaMemcpy(h_C, d_C, Row * Row * sizeof(float), cudaMemcpyDeviceToHost));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
