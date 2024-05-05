


#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include "error.h"
#include "GpuTimer.h"
#include "image_processing.cuh"


static std::string image_path = "woman.jpg";


void image_blur(ImageFilter filter_type) {
    switch (filter_type) {
        case AVERAGE:
            std::cout << "Image Filter: " << "AVERAGE" << std::endl;
            image_blur_average();
            break;
        case GAUSSIAN:
            std::cout << "Image Filter: " << "GAUSSIAN" << std::endl;
//            image_blur_gaussian();
            break;
        case MEDIAN:
            std::cout << "Image Filter: " << "MEDIAN" << std::endl;
//            image_blur_median();
            break;
        case SOBEL:
            std::cout << "Image Filter: " << "SOBEL" << std::endl;
//            image_blur_sobel();
            break;
        case LAPLACIAN:
            std::cout << "Image Filter: " << "LAPLACIAN" << std::endl;
//            image_blur_laplacian();
            break;
        case UNSHARP_MASK:
            std::cout << "Image Filter: " << "UNSHARP_MASK" << std::endl;
//            image_blur_unsharp_mask();
            break;
        default:
            std::cerr << "Error: " << "Unknown filter type" << std::endl;
            break;
    }

}


void image_blur_average() {




}