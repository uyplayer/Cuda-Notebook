/*
 * @Author: uyplayer
 * @Date: 2024/4/29 21:52
 * @Email: uyplayer@qq.com
 * @File: sumMatrixOnGPU_1D_grid_1D_block.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/professional-cuda-programming/examples
 * @Project_Name: Cuda-Notebook
 * @Description:
 */



#include <iostream>
#include <cuda_runtime.h>
#include "error.h"
#include "GpuTimer.h"
#include "Initializer.h"



// grid 2D block 1D
__global__ void sumMatrixOnGPU_2D_grid_1D_block_kernel(float *A, float *B, float *C, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

void sumMatrixOnGPU_2D_grid_1D_block() {

    std::cout << "Sum Matrix On GPU 1D Grid 1D Block" << std::endl;



}