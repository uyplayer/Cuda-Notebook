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




#define HANDLE_ERROR(a) { \
    if (a != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(a), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } else if (a == NULL) { \
        printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}


#define HANDLE_ERROR_CUDNN(a) { \
    if (a != CUDNN_STATUS_SUCCESS) { \
        printf("Error in %s at line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}


#define HANDLE_ERROR_CUBLAS(a) { \
    if (a != CUBLAS_STATUS_SUCCESS) { \
        printf("Error in %s at line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}


#define HANDLE_ERROR_CURAND(a) { \
    if (a != CURAND_STATUS_SUCCESS) { \
        printf("Error in %s at line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}


#endif //CUDA_NOTEBOOK_ERROR_H

