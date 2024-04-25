

#include "common.h"



// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output,int N)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    auto idz = blockIdx.z * blockDim.z + threadIdx.z;


    int globalIdx = idz * blockDim.x * blockDim.y * gridDim.x * gridDim.y +
                    idy * blockDim.x * gridDim.x +
                    idx;

    if (globalIdx < N)
    {

        output[globalIdx] = 2.0f * input[globalIdx];
    }
}
void clusters(){
    auto N = 50000;
    size_t size = N *  sizeof(float);

    float *input,*output = nullptr;

    HANDLE_ERROR(cudaMalloc((void**)&input, size));
    HANDLE_ERROR(cudaMalloc((void**)&output, size));


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<float> input_array(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        input_array[i] = static_cast<float>(dis(gen));
    }

    HANDLE_ERROR(cudaMemcpy(input, input_array.data(), size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output,N);

    // 等待 GPU 完成工作
    HANDLE_ERROR(cudaDeviceSynchronize());

    float *flatArrayC = new float[N];
    HANDLE_ERROR(cudaMemcpy(flatArrayC, output, size, cudaMemcpyDeviceToHost));
    std::cout << flatArrayC << std::endl;
    cudaFree(input);
    cudaFree(output);
    delete[] flatArrayC;

}