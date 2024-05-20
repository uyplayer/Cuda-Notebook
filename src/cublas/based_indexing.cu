//
// Created by uyplayer on 2024/5/20.
//


#include "error.h"
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>

//  Fortran风格的1-based索引和列主存储
// row ,col ,leading dimension(number of rows
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
// C风格的0-based索引和行主存储
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// row and col
#define M 6
#define N 5


// cuBLAS: 0-based indexing
static __inline__ void modify0(cublasHandle_t handle, float* m, int ldm, int n, int p, int q, float alpha, float beta)
{
    cublasSscal(handle, n - q + 1, &alpha, &m[IDX2F(p, q, ldm)], ldm);
    cublasSscal(handle, ldm - p + 1, &beta, &m[IDX2F(p, q, ldm)], 1);
}


// cuBLAS: 1-based indexing
void base_1()
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;


    float* devPtrA;
    float* a = 0;
    a = (float*)malloc(M * N * sizeof (*a));
    if (a == NULL)
    {
        std::cout << "host memory allocation failed" << std::endl;
        exit(1);
    }
    for (int j = 1; j <= N; j++)
    {
        for (int i = 1; i <= M; i++)
        {
            a[IDX2F(i, j, M)] = (float)(i + (j - 1) * M);
        }
    }

    cudaStat = cudaMalloc((void**)&devPtrA, M * N * sizeof(*a));
    if (cudaStat != cudaSuccess)
    {
        std::cout << "device memory allocation failed" << std::endl;
        free(a);
        exit(1);
    }
    // initialize handle
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed" << std::endl;
        cudaFree(devPtrA);
        free(a);
        exit(1);
    }
    // copy the array to the device
    stat = cublasSetMatrix(M,N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "data download failed" << std::endl;
        free(a);
        cudaFree(devPtrA);
        cublasDestroy(handle);
        exit(1);
    }

    modify0(handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
    stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "data download failed" << std::endl;
        free(a);
        cudaFree(devPtrA);
        cublasDestroy(handle);
        exit(1);
    }
    cudaFree(devPtrA);
    cublasDestroy(handle);
    for (int j = 1; j <= N; j++)
    {
        for (int i = 1; i <= M; i++)
        {
            std::cout << std::setw(7) << std::setprecision(0) << a[IDX2F(i,j,M)];
        }
        std::cout << std::endl;
    }
    free(a);
}

// cuBLAS: 0-based indexing
void base_0()
{
}
