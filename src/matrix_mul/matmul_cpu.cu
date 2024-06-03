//
// Created by uyplayer on 2024-06-03.
//


#include <iostream>
#include <chrono>
#include "matmul.cuh"


void matmul_cpu() {


    constexpr int row = 10000;
    constexpr int col = 5000;

    float *A = new float[row * col];
    float *B = new float[col * row];

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            A[i * col + j] = 1.0f;
        }
    }

    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            B[i * row + j] = 1.0f;
        }
    }
    int *C = new int[row * row];
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < row; j++) {
            C[i * row + j] = 0;
            for (int k = 0; k < col; k++) {
                C[i * row + j] += A[i * col + k] * B[k * row + j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s\n";

}