//
// Created by uyplayer on 2024-06-03.
//

#include <iostream>
#include "matmul.cuh"

int main() {

    std::cout << "Hello, matrix_mul !" << std::endl;
    matmul_cpu();
    matmul_gpu();
    return 0;

}