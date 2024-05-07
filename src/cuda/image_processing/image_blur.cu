


#include "image_blur.cuh"
#include <cuda_runtime.h>
#include "cuda_kernel.cuh"

ImageBlur::ImageBlur(const std::string &image_path) : Base(
        image_path) {
    std::cout << "Image path: " << image_path << std::endl;
    std::cout << "Image width: " << image_width << std::endl;
    std::cout << "Image height: " << image_height << std::endl;
    std::cout << "Image channels: " << image_channels << std::endl;
}


void ImageBlur::gaussian_kernel(int kernel_size) {
    if (kernel_size % 2 == 0) {
        kernel_size++;
    }

    const int rows = image_data.rows;
    const int cols = image_data.cols;


    // cuda data type uchar3
    uchar3 *d_input, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(uchar3));
    cudaMalloc(&d_output, rows * cols * sizeof(uchar3));
    // cuda data copy
    cudaMemcpy(d_input, image_data.ptr<uchar3>(), rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice);

    // set grid and  block size
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    imageBlurGaussianKernel<<<grid, block>>>(d_input, d_output, rows, cols, kernel_size);


    uchar3 *h_output = new uchar3[rows * cols];
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(uchar3), cudaMemcpyDeviceToHost);


    cv::Mat blurred_image = assembleImage(h_output, rows, cols);

    std::string filename = "src/cuda/image_processing/resource/gaussian_woman.jpg";
    cv::imwrite(filename, blurred_image);


    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;
}




void ImageBlur::median_kernel() {
    const int rows = image_data.rows;
    const int cols = image_data.cols;


    uchar3 *d_input, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(uchar3));
    cudaMalloc(&d_output, rows * cols * sizeof(uchar3));

    // cuda data copy
    cudaMemcpy(d_input, image_data.ptr<uchar3>(), rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice);

    // set grid and  block size
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    imageBlurMedianKernel<<<grid, block>>>(d_input, d_output, rows, cols);

    uchar3 *h_output = new uchar3[rows * cols];
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(uchar3), cudaMemcpyDeviceToHost);


    cv::Mat blurred_image = assembleImage(h_output, rows, cols);

    std::string filename = "src/cuda/image_processing/resource/median_woman.jpg";
    cv::imwrite(filename, blurred_image);


    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;



}

void ImageBlur::laplacian_kernel() {

    const int rows = image_data.rows;
    const int cols = image_data.cols;


    uchar3 *d_input, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(uchar3));
    cudaMalloc(&d_output, rows * cols * sizeof(uchar3));

    // cuda data copy
    cudaMemcpy(d_input, image_data.ptr<uchar3>(), rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice);

    // set grid and  block size
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    imageBlurLaplacianKernel<<<grid, block>>>(d_input, d_output, rows, cols);

    uchar3 *h_output = new uchar3[rows * cols];
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(uchar3), cudaMemcpyDeviceToHost);


    cv::Mat blurred_image = assembleImage(h_output, rows, cols);

    std::string filename = "src/cuda/image_processing/resource/laplacian_woman.jpg";
    cv::imwrite(filename, blurred_image);


    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;



}


cv::Mat ImageBlur::assembleImage(const uchar3 *data, int rows, int cols) {
    cv::Mat result(rows, cols, CV_8UC3);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            result.at<cv::Vec3b>(i, j)[0] = data[index].x;
            result.at<cv::Vec3b>(i, j)[1] = data[index].y;
            result.at<cv::Vec3b>(i, j)[2] = data[index].z;
        }
    }

    return result;
}



