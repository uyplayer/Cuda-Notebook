
#include "common.h"

static auto threadsPerBlock = 256;
static auto numElements = 1024;
static auto size = numElements * sizeof(float);


__global__ void dotProduct(const float *A, const float *B, float *C, int numElements) {

    __shared__ float temp[256];
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    auto threadId = threadIdx.x;

    if (index < numElements) {
        temp[threadId] = A[index] * B[index];
    } else {
        temp[threadId] = 0.0f;
    }

    // 结束的线程在这里等待其他线程完成
    __syncthreads();

    // 第一个线程进行计算总和
    if (threadId == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += temp[i];
        }
        atomicAdd(C, sum);
    }


}


float shared_memory() {

    auto *h_A = (float *) malloc(size);
    auto *h_B = (float *) malloc(size);
    auto h_C = 0.0f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    // 初始化输入数据
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(dis(gen));
        h_B[i] = static_cast<float>(dis(gen));
    }

    float *d_A, *d_B, *d_C;
    HANDLE_ERROR(cudaMalloc(&d_A, size));
    HANDLE_ERROR(cudaMalloc(&d_B, size));
    HANDLE_ERROR(cudaMalloc(&d_C, sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_C, &h_C, sizeof(float), cudaMemcpyHostToDevice));

    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    dotProduct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaMemcpy(&h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);


    printf("Dot Product: %f\n", h_C);

    free(h_A);
    free(h_B);

    HANDLE_ERROR(cudaFree(d_A));
    HANDLE_ERROR(cudaFree(d_B));
    HANDLE_ERROR(cudaFree(d_C));

    return h_C;

}


void run_shared_memory() {
    float result = shared_memory();
    if (result > 0) {
        printf("\nSuccess\n");
    } else {
        printf("\nFailed\n");
    }
}
