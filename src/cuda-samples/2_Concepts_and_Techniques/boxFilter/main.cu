#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "boxFilter.h"
#include "boxFilter.cuh"

using namespace std;
using namespace cv;

int main() {
    std::cout << "Hello, boxFilter !" << std::endl;
    std::string project_root{PROJECT_DIR};
    std::string imageFile = project_root + "/src/cuda-samples/2_Concepts_and_Techniques/boxFilter/img.png";

    Mat image = cv::imread(imageFile, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "打开文件失败" << std::endl;
        return -1;
    }
    // auto modified_image = boxFilterCpu(image, 10);
    auto modified_image = boxFilterGpu(image, 10);
    imshow("Original Image", image);
    imshow("Modified Image", modified_image);
    waitKey(0);
    return 0;
}
