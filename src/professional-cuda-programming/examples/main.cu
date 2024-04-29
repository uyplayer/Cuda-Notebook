

//
// Created by uyplayer on 2024/4/29.
//




#include <iostream>



extern void checkDeviceInfor();
extern void checkDimension();


int main() {
    std::cout << "Hello, CUDA!\n";

//    checkDeviceInfor();
    checkDimension();
    return 0;
}