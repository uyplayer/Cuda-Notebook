


#ifndef CUDA_NOTEBOOK_IMAGE_BLUR_H
#define CUDA_NOTEBOOK_IMAGE_BLUR_H


#include <iostream>
#include <cstdio>
#include "error.h"
#include "GpuTimer.h"
#include "image_processing.cuh"
#include "Base.h"

class ImageBlur : public Base {

    explicit ImageBlur(const std::string &image_path);

    ~ImageBlur() = default;

public:
    void average(int kernel_size = 5);

    void gaussian();

    void median();

    void laplacian();

    void unsharp_mask();


};

#endif //CUDA_NOTEBOOK_IMAGE_BLUR_H
