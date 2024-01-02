#include "common.h"

// 定义三维数组的宽度、高度和深度
const int width = 100;
const int height = 400;
const int depth = 200;

__global__ void dim3Kernel(const float *A, const float *B, float *C);

void dim3KernelCPU(const float *A, const float *B, float *C);

bool validateResults3(float *h_C, float *h_C_ref);

bool add3_dim();

void run_add3_dim() {
    bool result = add3_dim();
    if (result) {
        printf("\nSuccess\n");
    } else {
        printf("\nFailed\n");
    }
}

// add 3 dim
bool add3_dim() {
    // 计算数组大小
    size_t size = width * height * depth * sizeof(float);

    // 分配主机内存并初始化数组
    std::vector<std::vector<std::vector<float>>> arrayA(height, std::vector<std::vector<float>>(width, std::vector<float>(depth)));
    std::vector<std::vector<std::vector<float>>> arrayB(height, std::vector<std::vector<float>>(width, std::vector<float>(depth)));

    // 使用随机数初始化数组
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < depth; ++k) {
                arrayA[i][j][k] = static_cast<float>(dis(gen));
                arrayB[i][j][k] = static_cast<float>(dis(gen));
            }
        }
    }

    // 分配设备内存并将数据从主机拷贝到设备
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    HANDLE_ERROR(cudaMalloc((void**)&d_A, size));
    HANDLE_ERROR(cudaMalloc((void**)&d_B, size));
    HANDLE_ERROR(cudaMalloc((void**)&d_C, size));

    auto *flatArrayA = new float[width * height * depth];
    auto *flatArrayB = new float[width * height * depth];

    // 展平三维数组以用于 CUDA 操作
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < depth; ++k) {
                // 三维数组转换成一为数组
                int index = i * (width * depth) + j * depth + k;
                flatArrayA[index] = arrayA[i][j][k];
                flatArrayB[index] = arrayB[i][j][k];
            }
        }
    }
    HANDLE_ERROR(cudaMemcpy(d_A, flatArrayA, size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, flatArrayB, size, cudaMemcpyHostToDevice));

    // 定义线程块和网格大小
    // 每个线程块 10 * 10 * 5 = 500 线程
    // 一个重要的概念：nvidia GPU cuda 核函数中网格，线程块数量各有限制
    // 比如块 X 维度最大为 1024。Y 维度最大为 1024Z 维度最大为 64
    // 比如网格X 维度最大为 65535。Y 维度最大为 65535。Z 维度最大为 65535。
    dim3 threadsPerBlock(10, 10, 5); // 三维线程块
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

    dim3Kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // 等待 GPU 完成工作
    cudaDeviceSynchronize();

    auto *flatArrayC = new float[width * height * depth];
    // 将结果从设备拷贝回主机
    HANDLE_ERROR(cudaMemcpy(flatArrayC, d_C, size, cudaMemcpyDeviceToHost));
    auto *h_C_ref = new float[width * height * depth];
    dim3KernelCPU(flatArrayA, flatArrayB, h_C_ref);

    bool success = validateResults3(flatArrayC, h_C_ref);

    // 释放设备内存
    HANDLE_ERROR(cudaFree(d_A));
    HANDLE_ERROR(cudaFree(d_B));
    HANDLE_ERROR(cudaFree(d_C));

    // 释放主机内存
    delete[] flatArrayA;
    delete[] flatArrayB;
    delete[] flatArrayC;
    delete[] h_C_ref;

    return success;
}

__global__ void dim3Kernel(const float *A, const float *B, float *C) {
    auto xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    auto yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    auto zIndex = threadIdx.z + blockIdx.z * blockDim.z;

    if (xIndex < width && yIndex < height && zIndex < depth) {
        auto index = zIndex * width * height + yIndex * width + xIndex;
        C[index] = A[index] + B[index];
    }
}

void dim3KernelCPU(const float *A, const float *B, float *C) {
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                auto index = z * width * height + y * width + x;
                C[index] = A[index] + B[index];
            }
        }
    }
}

bool validateResults3(float *h_C, float *h_C_ref) {
    for (int i = 0; i < width * height * depth; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}
