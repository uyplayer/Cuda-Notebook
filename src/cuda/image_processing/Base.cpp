


#include "Base.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>



Base::Base(std::string image_path) : image_path(std::move(image_path)) {
    std::cout <<    "Base constructor" << std::endl;

}

void Base::check_opencv_package() {
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
    std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
    std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;
    std::cout << "OpenCV build information : " << cv::getBuildInformation() << std::endl;
    std::cout << "OpenCV Cuda support : " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
    std::cout << "OpenCV Cuda device : " << cv::cuda::getDevice() << std::endl;

}
