


#include <iostream>
#include <cstdio>
#include "error.h"
#include "GpuTimer.h"
#include "image_processing.cuh"
#include "Base.h"


class ImageBlur : public Base {

    explicit ImageBlur(const std::string &image_path) : Base(
            image_path) {
        std::cout << "Image path: " << image_path << std::endl;
        std::cout << "Image width: " << image_width << std::endl;
        std::cout << "Image height: " << image_height << std::endl;
        std::cout << "Image channels: " << image_channels << std::endl;
    }
    ~ImageBlur() = default;
    void average() {};
    void gaussian() {};
    void median() {};
    void laplacian() {};
    void unsharp_mask() {};


};


