//
// Created by uyplayer on 2024-06-19.
//




#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "error.h"


#define NUM_OF_BLOCKS 128
#define NUM_OF_THREADS 256


void generateInput(half2 *a, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        half2 temp;
        temp.x = static_cast<float>(rand() % 4);
        temp.y = static_cast<float>(rand() % 2);
        a[i] = temp;
    }
}

__forceinline__ __device__ void reduceInShared_intrinsics(half2 *const v) {
  if (threadIdx.x < 64)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 64]);
  __syncthreads();
  if (threadIdx.x < 32)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 32]);
  __syncthreads();
  if (threadIdx.x < 16)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 16]);
  __syncthreads();
  if (threadIdx.x < 8)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 8]);
  __syncthreads();
  if (threadIdx.x < 4)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 4]);
  __syncthreads();
  if (threadIdx.x < 2)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 2]);
  __syncthreads();
  if (threadIdx.x < 1)
    v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 1]);
  __syncthreads();
}


__forceinline__ __device__ void reduceInShared_native(half2 *const v) {
    if (threadIdx.x < 64) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 64];
    __syncthreads();
    if (threadIdx.x < 32) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 32];
    __syncthreads();
    if (threadIdx.x < 16) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 16];
    __syncthreads();
    if (threadIdx.x < 8) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 8];
    __syncthreads();
    if (threadIdx.x < 4) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 4];
    __syncthreads();
    if (threadIdx.x < 2) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 2];
    __syncthreads();
    if (threadIdx.x < 1) v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 1];
    __syncthreads();
}

__global__ void scalarProductKernel_native(half2 const *const a,
                                           half2 const *const b,
                                           float *const results,
                                           size_t const size) {
  const int stride = gridDim.x * blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  half2 value(0.f, 0.f);
  shArray[threadIdx.x] = value;

  for (int i = threadIdx.x + blockDim.x + blockIdx.x; i < size; i += stride) {
    value = a[i] * b[i] + value;
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_native(shArray);

  if (threadIdx.x == 0) {
    half2 result = shArray[0];
    float f_result = (float)result.y + (float)result.x;
    results[blockIdx.x] = f_result;
  }
}


__global__ void scalarProductKernel_intrinsics(half2 const *const a,
                                               half2 const *const b,
                                               float *const results,
                                               size_t const size) {
  const int stride = gridDim.x * blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  shArray[threadIdx.x] = __float2half2_rn(0.f);
  half2 value = __float2half2_rn(0.f);

  for (int i = threadIdx.x + blockDim.x + blockIdx.x; i < size; i += stride) {
    value = __hfma2(a[i], b[i], value);
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_intrinsics(shArray);

  if (threadIdx.x == 0) {
    half2 result = shArray[0];
    float f_result = __low2float(result) + __high2float(result);
    results[blockIdx.x] = f_result;
  }
}




int main() {
    std::cout << "Hello, fp16ScalarProduct !" << std::endl;

    // 使用BF16 数据类型
    size_t size = NUM_OF_BLOCKS * NUM_OF_THREADS * 16;
    half2 *vec[2];
    half2 *devVec[2];

    float *results;
    float *devResults;

    int device = 0;
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
    HANDLE_ERROR(cudaSetDevice(device));

    for (int i = 0; i < 2; ++i) {
        HANDLE_ERROR(cudaMallocHost((void **) &vec[i], size * sizeof *vec[i]));
        HANDLE_ERROR(cudaMalloc((void **) &devVec[i], size * sizeof *devVec[i]));

    }

    HANDLE_ERROR(cudaMallocHost((void **) &results, NUM_OF_BLOCKS * sizeof *results));
    HANDLE_ERROR(cudaMalloc((void **) &devResults, NUM_OF_BLOCKS * sizeof *devResults));

    for (int i = 0; i < 2; ++i) {
        generateInput(vec[i], size);
        HANDLE_ERROR(cudaMemcpy(devVec[i], vec[i], size * sizeof *vec[i], cudaMemcpyHostToDevice));
    }

    scalarProductKernel_native<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(devVec[0], devVec[1], devResults, size);

    HANDLE_ERROR(cudaMemcpy(results, devResults,
                            NUM_OF_BLOCKS * sizeof *results,
                            cudaMemcpyDeviceToHost));

    float result_native = 0;
    for (int i = 0; i < NUM_OF_BLOCKS; ++i) {
        result_native += results[i];
    }

    std::cout << "Result: " << result_native << std::endl;


    scalarProductKernel_intrinsics<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      devVec[0], devVec[1], devResults, size);

  HANDLE_ERROR(cudaMemcpy(results, devResults,
                             NUM_OF_BLOCKS * sizeof *results,
                             cudaMemcpyDeviceToHost));

  float result_intrinsics = 0;
  for (int i = 0; i < NUM_OF_BLOCKS; ++i) {
    result_intrinsics += results[i];
  }
  std::cout << "Result: " << result_intrinsics << std::endl;

  printf("p16ScalarProduct %s\n",
         (fabs(result_intrinsics - result_native) < 0.00001) ? "PASSED"
                                                             : "FAILED");

  for (int i = 0; i < 2; ++i) {
    HANDLE_ERROR(cudaFree(devVec[i]));
    HANDLE_ERROR(cudaFreeHost(vec[i]));
  }

  HANDLE_ERROR(cudaFree(devResults));
  HANDLE_ERROR(cudaFreeHost(results));

  return EXIT_SUCCESS;
}