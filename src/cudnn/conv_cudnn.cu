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

void cuda_dnn() {
    cudnnHandle_t cudnn;
    HANDLE_ERROR_CUDNN(cudnnCreate(&cudnn));



}





