

//
// Created by uyplayer on 2024/4/29.
//




#include <iostream>


extern void checkDeviceInfor();

extern void checkDimension();

extern void checkThreadIndex();

extern void defineGridBlock();

extern void sumArraysOnGPU_small_case();

extern void sumArraysOnGPU_timer();

int main() {
    std::cout << "Hello, CUDA!\n";

//    checkDeviceInfor();
//    checkDimension();
//    checkThreadIndex();
//    defineGridBlock();
//    sumArraysOnGPU_small_case();
//    sumArraysOnGPU_timer();
    return 0;
}