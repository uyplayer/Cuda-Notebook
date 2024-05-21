//
// Created by uyplayer on 2024/5/21.
//


#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>


void des_handle()
{
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "Failed to create handle" << std::endl;
        exit(1);
    }
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS destruction failed\n");
        std::cout << "Failed to destroy handle" << std::endl;
        exit(1);
    }
}


void addVector()
{
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

    int m = 2, n = 2, k = 2;
    float alpha = 1.0f;
    float beta = 0.0f;

    float h_A[] = {1.0, 2.0, 3.0, 4.0};
    float h_B[] = {5.0, 6.0, 7.0, 8.0};
    float h_C[] = {0.0, 0.0, 0.0, 0.0};

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // 执行矩阵乘法：C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    // 复制结果回主机
    cublasGetMatrix(m, n, sizeof(float), d_C, m, h_C, m);

    // 打印结果
    printf("Result matrix C:\n");
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            printf("%f ", h_C[i * n + j]);
        }
        printf("\n");
    }

    // 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}


void checkCublasStatus()
{
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        switch (status)
        {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("CUBLAS_STATUS_ALLOC_FAILED\n");
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            printf("CUBLAS_STATUS_INVALID_VALUE\n");
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("CUBLAS_STATUS_ARCH_MISMATCH\n");
            break;
        case CUBLAS_STATUS_MAPPING_ERROR:
            printf("CUBLAS_STATUS_MAPPING_ERROR\n");
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("CUBLAS_STATUS_EXECUTION_FAILED\n");
            break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            printf("CUBLAS_STATUS_INTERNAL_ERROR\n");
            break;
        default:
            printf("Unknown error\n");
        }
    }
    cublasDestroy(handle);
}
