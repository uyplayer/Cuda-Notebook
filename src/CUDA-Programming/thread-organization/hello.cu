

//
// Created by uyplayer on 2024/4/28.
//


#include <stdio.h>


__global__ void hello2() {
    printf("Hello, world from block %d, thread %d!\n", blockIdx.x, threadIdx.x);
}


__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}


__global__ void hello_from_gpu_2()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}


__global__ void hello_from_gpu_3()
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}