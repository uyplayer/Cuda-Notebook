/*
 * @Author: uyplayer
 * @Date: 2024/5/4 10:56
 * @Email: uyplayer@qq.com
 * @File: main.cpp
 * @Software: CLion
 * @Dir: Cuda-Notebook / src/cuda/image_processing
 * @Project_Name: Cuda-Notebook
 * @Description:
 */


#include <iostream>
#include "cjson/cJSON.h"
#include "image_processing.cuh"


extern void image_blur(ImageFilter);


int main(){

    std::cout << "Hello image_processing" << std::endl;
//    image_blur(AVERAGE);
    return 0;
}