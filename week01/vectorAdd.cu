
#include "../common/common.h"



__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements);

void vectorAddCPU(const float *A, const float *B, float *C, int numElements);

bool validateResults(float *h_C, float *h_C_ref, int numElements);

bool runVectorAddition();

void runVectorAdditionResult() {
    if (runVectorAddition()) {
        printf("\nSuccess\n");
    } else {
        printf("\nFailed\n");
    }
}

/// 运行相加操作
/// 运行相加操作，并返回计算是否成功
bool runVectorAddition() {
    printf("runVectorAddition running for test\n");

    // 元素个数
    int numElements = 50000;
    // 元素大小
    size_t size = numElements * sizeof(float);

    // 分配主机内存
    auto *h_A = new float[numElements];
    auto *h_B = new float[numElements];
    auto *h_C = new float[numElements];

    // 随机化
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    // 初始化
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(dis(gen));
        h_B[i] = static_cast<float>(dis(gen));
    }

    // 分配设备内存
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    HANDLE_ERROR(cudaMalloc((void **) &d_A, size));
    HANDLE_ERROR(cudaMalloc((void **) &d_B, size));
    HANDLE_ERROR(cudaMalloc((void **) &d_C, size));

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // 执行核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // 等待GPU完成工作
    cudaDeviceSynchronize();

    // 将结果从设备拷贝回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 使用CPU进行向量加法进行验证
    auto *h_C_ref = new float[numElements];
    vectorAddCPU(h_A, h_B, h_C_ref, numElements);

    // 验证结果
    bool success = validateResults(h_C, h_C_ref, numElements);

    // 释放设备内存
    HANDLE_ERROR(cudaFree(d_A));
    HANDLE_ERROR(cudaFree(d_B));
    HANDLE_ERROR(cudaFree(d_C));

    // 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return success;
}

/// 验证相加操作
/// \param h_C
/// \param h_C_ref
/// \param numElements
/// \return 相加是否成功true false
bool validateResults(float *h_C, float *h_C_ref, int numElements) {
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

/// 核函数，进行计算操作；所有的核函数返回值void
/// \param A 要加的向量
/// \param B 要加的向量
/// \param C 相加结果
/// \param numElements 相加元素的数量
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    /*
     * threadIdx是一个uint3类型，表示一个线程的索引。
     * blockIdx是一个uint3类型，表示一个线程块的索引，一个线程块中通常有多个线程。
     * blockDim是一个dim3类型，表示线程块的大小。
     * gridDim是一个dim3类型，表示网格的大小，一个网格中通常有多个线程块。
     * */

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}


/// 普通cpu执行函数
/// \param A 要加的向量
/// \param B 要加的向量
/// \param C 相加结果
/// \param numElements 相加元素的数量
void vectorAddCPU(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; ++i) {
        C[i] = A[i] + B[i];
    }
}


