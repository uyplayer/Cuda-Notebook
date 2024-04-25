/*
 * @Author: uyplayer
 * @Date: 2024/4/25 18:31
 * @Email: uyplayer@qq.com
 * @File: error.h
 * @Software: CLion
 * @Dir: Cuda-Notebook / common
 * @Project_Name: Cuda-Notebook
 * @Description:
 */



#ifndef CUDA_NOTEBOOK_ERROR_H
#define CUDA_NOTEBOOK_ERROR_H



#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <random>
#include "cudnn.h"



static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL(a) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


#endif

#endif //CUDA_NOTEBOOK_ERROR_H
