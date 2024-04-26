

//
// Created by uyplayer on 2024/4/26.
//



#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include "error.h"




__global__ void vectorAdd(int *a, int *b, int *c, int N){
    // Calculate global thread thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(tid < N){
        c[tid] = a[tid] + b[tid];
    }
}

void verify_result(int *a, int *b, int *c, int N) {
    for (int i = 0; i < N; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main(){
    std::cout << "Hello, vectorAdd_pinned !" << std::endl;



    constexpr int N = 1 << 20;   // 1M elements   size of 2^16
    constexpr size_t bytes = N * sizeof(float);

    // Declare unified memory pointerss
    int *h_a, *h_b, *h_c;
    HANDLE_ERROR(cudaMallocHost(&h_a, bytes));
    HANDLE_ERROR(cudaMallocHost(&h_b, bytes));
    HANDLE_ERROR(cudaMallocHost(&h_c, bytes));

    // Initialize vectors
    for(int i = 0; i < N; i++){
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }


    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    HANDLE_ERROR(cudaMalloc(&d_a, bytes));
    HANDLE_ERROR(cudaMalloc(&d_b, bytes));
    HANDLE_ERROR(cudaMalloc(&d_c, bytes));

    // Copy data from the host to the device (CPU -> GPU)
    HANDLE_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    // Threads per CTA (1024 threads per CTA)
    int NUM_THREADS = 1 << 10;

    int num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;

    // Launch the kernel on the GPU
    vectorAdd<<<num_blocks, NUM_THREADS>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    // synchronization
    // barrier.
    verify_result(h_a, h_b, h_c, N);

    // Free pinned memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;

}