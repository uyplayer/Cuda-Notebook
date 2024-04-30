/*
 * @Author: uyplayer
 * @Date: 2024/4/29 18:40
 * @Email: uyplayer@qq.com
 * @File: sumArraysOnGPU-small-case.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/professional-cuda-programming/examples
 * @Project_Name: Cuda-Notebook
 * @Description:
 */


#include <iostream>
#include <cuda_runtime.h>
#include "error.h"
#include "GpuTimer.h"



void initialData(float *ip, int size) {

    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }

}


void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


void sumArraysOnGPU_small_case() {

    std::cout << "sumArraysOnGPU_small_case" << std::endl;

    int dev = 0;
    CHECK(cudaSetDevice(dev));

    int nElem = 1 << 10;
    std::cout << "Vector size " << nElem << std::endl;
    size_t nBytes = nElem * sizeof(float);
    std::cout << "nBytes size " << nBytes << " bytes" << std::endl;

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = new float[nElem];
    h_B = new float[nElem];
    hostRef = new float[nElem];
    gpuRef = new float[nElem];

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);


    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **) &d_A, nBytes));
    CHECK(cudaMalloc((float **) &d_B, nBytes));
    CHECK(cudaMalloc((float **) &d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    dim3 block(nElem);
    dim3 grid(1);


    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    printf("Execution configure <<<%d, %d>>>\n", grid.x, block.x);

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());


}