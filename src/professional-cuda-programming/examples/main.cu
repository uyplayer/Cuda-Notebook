

//
// Created by uyplayer on 2024/4/29.
//




#include <iostream>



extern void checkDeviceInfor();
extern void checkDimension();
extern void checkThreadIndex();

int main() {
    std::cout << "Hello, CUDA!\n";

//    checkDeviceInfor();
//    checkDimension();
    checkThreadIndex();
    return 0;
}