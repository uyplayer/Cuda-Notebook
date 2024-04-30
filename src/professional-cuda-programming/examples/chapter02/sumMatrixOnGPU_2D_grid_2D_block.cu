

//
// Created by uyplayer on 2024/4/30.
//



#include <iostream>
#include <cuda_runtime.h>
#include "error.h"
#include "GpuTimer.h"
#include "Initializer.h"

// grid 2D block 2D
__global__ void sumMatrixOnGPU_2D_grid_2D_block_kernel(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}