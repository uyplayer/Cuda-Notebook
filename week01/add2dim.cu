

#include "../common/common.h"


// 定义二维数组的宽度和高度
const int width = 1000;
const int height = 40000;

__global__ void dim2Kernel(const float *A, const float *B, float *C);

void dim2KernelCPU(const float **A, const float **B, float *C);

bool validateResults(float *h_C, float *h_C_ref);

bool add2_dim();

void run_add2_dim() {
    if (add2_dim()) {
        printf("\nSuccess\n");
    } else {
        printf("\nFailed\n");
    }
}

// add 2 dim
bool add2_dim() {
    printf("add2_dim running for test\n");
    // 计算数组大小
    size_t size = width * height * sizeof(float);

    // 分配主机内存并初始化数组
    auto **arrayA = new float *[height];
    auto **arrayB = new float *[height];
    for (int i = 0; i < height; ++i) {

        arrayA[i] = new float[width];
        arrayB[i] = new float[width];

    }

    // 使用随机数初始化数组
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            arrayA[i][j] = static_cast<float>(dis(gen));
            arrayB[i][j] = static_cast<float>(dis(gen));
        }
    }

    // 分配设备内存并将数据从主机拷贝到设备
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    HANDLE_ERROR(cudaMalloc((void **) &d_A, size));
    HANDLE_ERROR(cudaMalloc((void **) &d_B, size));
    HANDLE_ERROR(cudaMalloc((void **) &d_C, size));

    auto * flatArrayA = new float[width * height];
    auto * flatArrayB = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            flatArrayA[i * width + j] = arrayA[i][j];
            flatArrayB[i * width + j] = arrayB[i][j];
        }
    }

    cudaMemcpy(d_A, flatArrayA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatArrayB, size, cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);


    dim2Kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);


    // 等待 GPU 完成工作
    cudaDeviceSynchronize();
    auto *arrayC = new float[width * height];
    // 将结果从设备拷贝回主机
    cudaMemcpy(arrayC, d_C, size, cudaMemcpyDeviceToHost);

    auto *h_C_ref = new float[width * height];
    dim2KernelCPU((const float **) arrayA, (const float **) arrayB, h_C_ref);


    // 验证结果
    bool success = validateResults(arrayC, h_C_ref);

    // 释放设备内存
    HANDLE_ERROR(cudaFree(d_A));
    HANDLE_ERROR(cudaFree(d_B));
    HANDLE_ERROR(cudaFree(d_C));

    // 释放主机内存
    delete[] flatArrayA;
    delete[] flatArrayB;
    delete[] arrayA;
    delete[] arrayB;
    delete[] arrayC;
    delete[] h_C_ref;


    return success;
}

__global__ void dim2Kernel(const float *A, const float *B, float *C) {
    auto xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    auto yIndex = threadIdx.y + blockIdx.y * blockDim.y;

    if (xIndex < width && yIndex < height) {
        auto index = yIndex * width + xIndex;
        C[index] = A[index] + B[index];
    }
}


void dim2KernelCPU(const float **A, const float **B, float *C) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            C[index] = A[y][x] + B[y][x];
        }
    }
}

bool validateResults(float *h_C, float *h_C_ref) {
    for (int i = 0; i < width * height; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}
