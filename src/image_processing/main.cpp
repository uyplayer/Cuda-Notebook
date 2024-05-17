


#include <iostream>
#include "image_processing.cuh"
#include "image_blur.cuh"
#include <opencv2/opencv.hpp>




int main(){
    std::cout << "Hello image_processing" << std::endl;
    std::string image_path = "src/cuda/image_processing/resource/woman.jpg";

    ImageBlur image_blur{image_path};

    image_blur.gaussian_kernel(20);
    image_blur.median_kernel();
    image_blur.laplacian_kernel();

    return 0;
}

