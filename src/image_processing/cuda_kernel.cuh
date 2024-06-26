

//
// Created by uyplayer on 2024/5/6.
//

#ifndef CUDA_NOTEBOOK_CUDA_KERNEL_CUH
#define CUDA_NOTEBOOK_CUDA_KERNEL_CUH


#include <cuda_runtime.h>

__global__ void imageBlurGaussianKernel(const uchar3* input, uchar3* output, int rows, int cols, int kernel);


__global__ void imageBlurMedianKernel(const uchar3* input, uchar3* output, int rows, int cols);


__global__ void imageBlurLaplacianKernel(const uchar3* input, uchar3* output, int rows, int cols);



#endif //CUDA_NOTEBOOK_CUDA_KERNEL_CUH
