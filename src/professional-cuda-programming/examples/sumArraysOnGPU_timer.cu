/*
 * @Author: uyplayer
 * @Date: 2024/4/29 19:53
 * @Email: uyplayer@qq.com
 * @File: sumArraysOnGPU_timer.cu
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/professional-cuda-programming/examples
 * @Project_Name: Cuda-Notebook
 * @Description:
 */


#include <iostream>
#include <cuda_runtime.h>
#include "error.h"
#include <GpuTimer.h>
#include <Initializer.h>


__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}


void sumArraysOnHost(float *A, float *B, float *C, const int N) {

    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }

}


void sumArraysOnGPU_timer() {

    std::cout << "Sum Arrays On GPU Timer" << std::endl;


    int dev{0};
    cudaDeviceProp deviceProp{};
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    std::cout << "Using Device " << dev << ": " << deviceProp.name << std::endl;
    CHECK(cudaSetDevice(dev));

    int nElem = 1 << 24;
    std::cout << "Vector size " << nElem << std::endl;

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);


    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = new float[nElem];
    h_B = new float[nElem];

    hostRef = (float *) malloc(nBytes);
    gpuRef = (float *) malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    iStart = seconds();

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    std::cout << "initialData Time elapsed " << iElaps << " sec" << std::endl;
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    std::cout << "sumArraysOnHost Time elapsed " << iElaps << " sec" << std::endl;

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **) &d_A, nBytes));
    CHECK(cudaMalloc((float **) &d_B, nBytes));
    CHECK(cudaMalloc((float **) &d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));


    int iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    iStart = seconds();
    // kernel
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);

    // check kernel error
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

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

}