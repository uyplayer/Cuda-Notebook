


#ifndef CUDA_NOTEBOOK_BASE_H
#define CUDA_NOTEBOOK_BASE_H

#include <iostream>
#include <opencv2/core/mat.hpp>

class Base {
public:
    explicit Base(std::string image_path);

    ~Base() = default;

    Base(const Base &base) = delete;

    static void check_opencv_package();

private:

protected:
    cv::Mat image_data;
    std::string image_path;
    int image_width;
    int image_height;
    int image_channels;
};


#endif //CUDA_NOTEBOOK_BASE_H
