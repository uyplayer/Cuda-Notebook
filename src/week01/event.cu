


#include "common.h"



__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}


bool event_cuda(){
    const auto numElements = 50000;
    auto size = numElements * sizeof(float);

    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C(numElements);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(dis(gen));
        h_B[i] = static_cast<float>(dis(gen));
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // 分配设备端内存
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 复制数据从主机到设备端
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);


    // 创建事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 开始就记录
    cudaEventRecord(start);

    // 启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // 记录结束了
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Vector Add took " << milliseconds << " milliseconds\n";

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 复制结果从设备到主机端内存
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Test PASSED\n";

    // 重置设备
    cudaDeviceReset();

    return true;

}