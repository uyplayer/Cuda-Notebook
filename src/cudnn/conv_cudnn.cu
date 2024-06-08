#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include "error.h"

__global__ void dev_const(float *px, float k) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = k;
}

__global__ void dev_iota(float *px) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = tid;
}

void print(const float *data, int n, int c, int h, int w) {
    std::vector<float> buffer(1 << 20);
    HANDLE_ERROR(cudaMemcpy(
                buffer.data(), data,
                n * c * h * w * sizeof(float),
                cudaMemcpyDeviceToHost));
    int a = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
            for (int k = 0; k < h; ++k) {
                for (int l = 0; l < w; ++l) {
                    std::cout << std::setw(4) << std::right << buffer[a];
                    ++a;
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

void cuda_dnn() {
    cudnnHandle_t cudnn;
    HANDLE_ERROR_CUDNN(cudnnCreate(&cudnn));

    // input
    const int in_n = 1;
    const int in_c = 1;
    const int in_h = 5;
    const int in_w = 5;
    std::cout << "in_n: " << in_n << std::endl;
    std::cout << "in_c: " << in_c << std::endl;
    std::cout << "in_h: " << in_h << std::endl;
    std::cout << "in_w: " << in_w << std::endl;
    std::cout << std::endl;

    cudnnTensorDescriptor_t in_desc;
    HANDLE_ERROR_CUDNN(cudnnCreateTensorDescriptor(&in_desc));
    HANDLE_ERROR_CUDNN(cudnnSetTensor4dDescriptor(
                in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                in_n, in_c, in_h, in_w));

    float *in_data;
    HANDLE_ERROR(cudaMalloc(
                &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

    // filter
    const int filt_k = 1;
    const int filt_c = 1;
    const int filt_h = 2;
    const int filt_w = 2;
    std::cout << "filt_k: " << filt_k << std::endl;
    std::cout << "filt_c: " << filt_c << std::endl;
    std::cout << "filt_h: " << filt_h << std::endl;
    std::cout << "filt_w: " << filt_w << std::endl;
    std::cout << std::endl;

    cudnnFilterDescriptor_t filt_desc;
    HANDLE_ERROR_CUDNN(cudnnCreateFilterDescriptor(&filt_desc));
    HANDLE_ERROR_CUDNN(cudnnSetFilter4dDescriptor(
                filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                filt_k, filt_c, filt_h, filt_w));

    float *filt_data;
    HANDLE_ERROR(cudaMalloc(
            &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

    // convolution
    const int pad_h = 1;
    const int pad_w = 1;
    const int str_h = 1;
    const int str_w = 1;
    const int dil_h = 1;
    const int dil_w = 1;
    std::cout << "pad_h: " << pad_h << std::endl;
    std::cout << "pad_w: " << pad_w << std::endl;
    std::cout << "str_h: " << str_h << std::endl;
    std::cout << "str_w: " << str_w << std::endl;
    std::cout << "dil_h: " << dil_h << std::endl;
    std::cout << "dil_w: " << dil_w << std::endl;
    std::cout << std::endl;

    cudnnConvolutionDescriptor_t conv_desc;
    HANDLE_ERROR_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    HANDLE_ERROR_CUDNN(cudnnSetConvolution2dDescriptor(
                conv_desc,
                pad_h, pad_w, str_h, str_w, dil_h, dil_w,
                CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // output
    int out_n;
    int out_c;
    int out_h;
    int out_w;

    HANDLE_ERROR_CUDNN(cudnnGetConvolution2dForwardOutputDim(
                conv_desc, in_desc, filt_desc,
                &out_n, &out_c, &out_h, &out_w));

    std::cout << "out_n: " << out_n << std::endl;
    std::cout << "out_c: " << out_c << std::endl;
    std::cout << "out_h: " << out_h << std::endl;
    std::cout << "out_w: " << out_w << std::endl;
    std::cout << std::endl;

    cudnnTensorDescriptor_t out_desc;
    HANDLE_ERROR_CUDNN(cudnnCreateTensorDescriptor(&out_desc));
    HANDLE_ERROR_CUDNN(cudnnSetTensor4dDescriptor(
                out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                out_n, out_c, out_h, out_w));

    float *out_data;
    HANDLE_ERROR(cudaMalloc(
                &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

    // algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int returned_algo_count;
    HANDLE_ERROR_CUDNN(cudnnFindConvolutionForwardAlgorithm(
                cudnn,
                in_desc, filt_desc, conv_desc, out_desc,
                1, &returned_algo_count, &algo_perf));

    cudnnConvolutionFwdAlgo_t algo = algo_perf.algo;

    std::cout << "Convolution algorithm: " << algo << std::endl;
    std::cout << std::endl;

    // workspace
    size_t ws_size;
    HANDLE_ERROR_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

    float *ws_data;
    HANDLE_ERROR(cudaMalloc(&ws_data, ws_size));

    std::cout << "Workspace size: " << ws_size << std::endl;
    std::cout << std::endl;

    // perform
    float alpha = 1.f;
    float beta = 0.f;
    dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
    dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);
    HANDLE_ERROR_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha, in_desc, in_data, filt_desc, filt_data,
            conv_desc, algo, ws_data, ws_size,
            &beta, out_desc, out_data));

    // results
    std::cout << "in_data:" << std::endl;
    print(in_data, in_n, in_c, in_h, in_w);

    std::cout << "filt_data:" << std::endl;
    print(filt_data, filt_k, filt_c, filt_h, filt_w);

    std::cout << "out_data:" << std::endl;
    print(out_data, out_n, out_c, out_h, out_w);

    // finalizing
    HANDLE_ERROR(cudaFree(ws_data));
    HANDLE_ERROR(cudaFree(out_data));
    HANDLE_ERROR_CUDNN(cudnnDestroyTensorDescriptor(out_desc));
    HANDLE_ERROR_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    HANDLE_ERROR(cudaFree(filt_data));
    HANDLE_ERROR_CUDNN(cudnnDestroyFilterDescriptor(filt_desc));
    HANDLE_ERROR(cudaFree(in_data));
    HANDLE_ERROR_CUDNN(cudnnDestroyTensorDescriptor(in_desc));
    HANDLE_ERROR_CUDNN(cudnnDestroy(cudnn));
}
