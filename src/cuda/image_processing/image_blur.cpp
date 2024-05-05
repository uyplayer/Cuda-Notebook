


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
    auto data = image_data.data;
    for (int i = 0; i < image_data.rows; ++i) {
        for (int j = 0; j < image_data.cols; ++j) {
            for (int k = 0; k < image_data.channels(); ++k) {
                auto elem = static_cast<float>(data[i * image_data.cols * image_data.channels() + j * image_data.channels() + k]);
                std::cout << elem << " ";
            }
        }
    }
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


