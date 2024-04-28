

//
// Created by uyplayer on 2024/4/28.
//





#include <iostream>

extern void add_1();

extern void add2gpu();

int main() {

    std::cout << "Hello, prerequisites-for-speedup!" << std::endl;
    add_1();
    add2gpu();
    return 0;

}