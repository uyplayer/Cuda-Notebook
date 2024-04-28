

//
// Created by uyplayer on 2024/4/28.
//



#include <iostream>


extern void main_add_basic();
extern void main_add1_cuda();

int main() {

    std::cout << "Hello basic-framework" << std::endl;
//    main_add_basic();
    main_add1_cuda();
    return 0;
}