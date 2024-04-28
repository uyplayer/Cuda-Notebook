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


#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define HANDLE_ERROR(a) { \
    if (a != cudaSuccess) { \
        std::cout << cudaGetErrorString(a) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define HANDLE_ERROR_CUDNN(a) { \
    if (a != CUDNN_STATUS_SUCCESS) { \
        std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define HANDLE_ERROR_CUBLAS(a) { \
    if (a != CUBLAS_STATUS_SUCCESS) { \
        std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define HANDLE_ERROR_CURAND(a) { \
    if (a != CURAND_STATUS_SUCCESS) { \
        std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        std::cout << "CUDA Error:" << std::endl;      \
        std::cout << "    File:       " << __FILE__ << std::endl;     \
        std::cout << "    Line:       " << __LINE__ << std::endl;     \
        std::cout << "    Error code: " << error_code << std::endl;   \
        std::cout << "    Error text: " << cudaGetErrorString(error_code) << std::endl; \
        exit(1);                                      \
    }                                                 \
} while (0)

#endif //CUDA_NOTEBOOK_ERROR_H

