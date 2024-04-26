#include <iostream>
#include <cuda_runtime.h>
#include "error.h"
#include "GpuTimer.h"

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

void add_value() {
    int result;

    int* d_result;
    HANDLE_ERROR(cudaMalloc(&d_result, sizeof(int)));

    add<<<1, 1>>>(100, 100, d_result);

    HANDLE_ERROR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost)); // 将计算结果从设备复制到主机内存

    cudaFree(d_result);



    std::cout << "result = " << result << std::endl;
    std::cout << "add_value() done" << std::endl;
}
