
#include "common.h"

// cuda 常量内存只能放在外面，因为这欸内存是在cpu初始化然后复制到gpu。也是在直接在核函数里面使用
__constant__ float constVector[256];

__global__ void vectorAddConstant(const float *input, float *output, int n) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = input[idx] + constVector[idx];
    }
}



bool const_cuda(){
    const int numElements = 256;
    size_t size = numElements * sizeof(float);

    // 分配和初始化主机端内存
    float hostInput[numElements], hostOutput[numElements], hostConst[numElements];
    for (int i = 0; i < numElements; ++i) {
        hostInput[i] = float(i);
        hostConst[i] = float(i);
    }

    // 分配设备内存
    float *devInput, *devOutput;
    cudaMalloc((void **)&devInput, size);
    cudaMalloc((void **)&devOutput, size);

    // 将常量和输入数据复制到设备
    cudaMemcpyToSymbol(constVector, hostConst, size);
    cudaMemcpy(devInput, hostInput, size, cudaMemcpyHostToDevice);

    // 启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddConstant<<<blocksPerGrid, threadsPerBlock>>>(devInput, devOutput, numElements);

    // 将结果从设备复制回主机
    cudaMemcpy(hostOutput, devOutput, size, cudaMemcpyDeviceToHost);

    // 检查结果
    for (int i = 0; i < numElements; ++i) {
        if (fabs(hostInput[i] + hostConst[i] - hostOutput[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED\n";

    // 清理
    cudaFree(devInput);
    cudaFree(devOutput);
    return true;
}



void run_const_memory() {
    bool result = const_cuda();
    printf("\nSuccess\n");
}
