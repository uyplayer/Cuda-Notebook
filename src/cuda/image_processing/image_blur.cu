


#include <iostream>
#include <cstdio>
#include "error.h"
#include "GpuTimer.h"
#include "image_processing.cuh"
#include "Base.h"


class ImageBlur : public Base {

    explicit ImageBlur(const std::string &image_path, const std::string &output_path, int kernel_size) : Base(
            image_path) {
        std::cout << "ImageBlur constructor" << std::endl;
    }
    ~ImageBlur() = default;
    void average() {};
    void gaussian() {};
    void median() {};
    void laplacian() {};
    void unsharp_mask() {};


};


