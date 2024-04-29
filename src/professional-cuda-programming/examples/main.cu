

//
// Created by uyplayer on 2024/4/29.
//




#include <iostream>



extern void checkDeviceInfor();


int main() {
    std::cout << "Hello, CUDA!\n";

    checkDeviceInfor();

    return 0;
}