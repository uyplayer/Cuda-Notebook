

//
// Created by uyplayer on 2024/4/29.
//



#include <iostream>
#include <stdio.h>
#include "error.h"


__global__ void checkIndex(void) {
    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

}


void checkDimension() {

    constexpr int N{1 << 20};

    dim3 block(64);
    dim3 grid((N + block.x - 1) / block.x);


    std::cout << "grid.x " << grid.x << " grid.y " << grid.y << " grid.z " << grid.z << std::endl;
    std::cout << "block.x " << block.x << " block.y " << block.y << " block.z " << block.z << std::endl;

    checkIndex<<<grid, block>>>();

    CHECK(cudaDeviceReset());

}