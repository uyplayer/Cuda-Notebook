


#ifndef CUDA_NOTEBOOK_IMAGE_BLUR_CUH
#define CUDA_NOTEBOOK_IMAGE_BLUR_CUH


#include <iostream>
#include <cstdio>
#include "error.h"
#include "GpuTimer.h"
#include "image_processing.cuh"
#include "Base.h"

class ImageBlur : public Base {

public:
    void average(int kernel_size = 5);

    void gaussian();

    void median();

    void laplacian();

    void unsharp_mask();

    static cv::Mat  assembleImage(const uchar3* data, int rows, int cols);
    explicit ImageBlur(const std::string &image_path);

    ~ImageBlur() = default;
};

#endif //CUDA_NOTEBOOK_IMAGE_BLUR_CUH
