//
// Created by uyplayer on 2024-06-30.
//

#ifndef BOXFILTER_CUH
#define BOXFILTER_CUH

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>


__global__ void boxFilterKernel(const uchar* input, uchar* output, int width, int height, int channels, int kernelSize, size_t pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int halfKernel = kernelSize / 2;
    if (x >= width || y >= height || c >= channels) {
        return;
    };

    int sum = 0;
    int count = 0;
    for (int ky = -halfKernel; ky <= halfKernel; ky++) {
        for (int kx = -halfKernel; kx <= halfKernel; kx++) {
            int nx = x + kx;
            int ny = y + ky;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[(ny * pitch + nx * channels) + c];
                count++;
            }
        }
    }
    output[(y * pitch + x * channels) + c] = sum / count;
}

cv::Mat boxFilterGpu(const cv::Mat &input, int kernelSize) {
    std::cout << "Using GPU" << std::endl;
    if (kernelSize % 2 == 0) {
        kernelSize++;
    }

    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t pitch;
    uchar *d_input, *d_output;
    cudaMallocPitch(&d_input, &pitch, width * channels, height);
    cudaMallocPitch(&d_output, &pitch, width * channels, height);
    cudaMemcpy2D(d_input, pitch, input.ptr(), input.step, width * channels, height, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, channels);
    boxFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels, kernelSize, pitch);
    cv::Mat output(input.size(), input.type());
    cudaMemcpy2D(output.ptr(), output.step, d_output, pitch, width * channels, height, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

#endif //BOXFILTER_CUH
