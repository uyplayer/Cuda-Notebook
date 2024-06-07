//
// Created by uyplayer on 2024-06-06.
//


#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include "error.h"


int main() {
    cudnnHandle_t cudnn;
    HANDLE_ERROR_CUDNN(cudnnCreate(&cudnn));

    // input
    constexpr int in_n = 1;
    constexpr int inc_c = 1;
    constexpr int in_h = 5;
    constexpr int in_w = 5;

    std::cout << "in_n: " << in_n << std::endl;
    std::cout << "inc_c: " << inc_c << std::endl;
    std::cout << "in_h: " << in_h << std::endl;
    std::cout << "in_w: " << in_w << std::endl;

    cudnnTensorDescriptor_t in_desc;
    HANDLE_ERROR_CUDNN(cudnnCreateTensorDescriptor(&in_desc));
HANDLE_ERROR_CUDNN(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, inc_c, in_h, in_w));



}


































