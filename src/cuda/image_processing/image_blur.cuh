


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
    explicit ImageBlur(const std::string &image_path);

    void gaussian_kernel(int kernel_size = 5);
    void median_kernel();

    void laplacian();

    void unsharp_mask();

    static cv::Mat assembleImage(const uchar3 *data, int rows, int cols);


    ~ImageBlur() = default;
};

#endif //CUDA_NOTEBOOK_IMAGE_BLUR_CUH
