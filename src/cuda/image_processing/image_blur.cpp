


#include "image_blur.h"
#include <cuda_runtime.h>

ImageBlur::ImageBlur(const std::string &image_path) : Base(
        image_path) {
    std::cout << "Image path: " << image_path << std::endl;
    std::cout << "Image width: " << image_width << std::endl;
    std::cout << "Image height: " << image_height << std::endl;
    std::cout << "Image channels: " << image_channels << std::endl;
}


void ImageBlur::average(int kernel_size) {
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




}


void ImageBlur::gaussian() {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
    cv::imwrite("gaussian.jpg", result);
}

void ImageBlur::median() {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat result;
    cv::medianBlur(image, result, 5);
    cv::imwrite("median.jpg", result);
}

void ImageBlur::laplacian() {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat result;
    cv::Laplacian(image, result, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::imwrite("laplacian.jpg", result);
}

void ImageBlur::unsharp_mask() {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(5, 5), 1.5);
    cv::addWeighted(image, 1.5, result, -0.5, 0, result);
    cv::imwrite("unsharp_mask.jpg", result);
}


