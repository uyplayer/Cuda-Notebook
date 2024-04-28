

//
// Created by uyplayer on 2024/4/28.
//




#include <iostream>


extern __global__ void hello2();
extern __global__ void hello_from_gpu();
extern __global__ void hello_from_gpu_2();
extern __global__ void hello_from_gpu_3();

int main(){


    std::cout << "Hello thread-organization" << std::endl;

    hello2<<<1, 1>>>();
    hello_from_gpu<<<2, 4>>>();
    hello_from_gpu_2<<<2, 4>>>();
    hello_from_gpu_3<<<2, 4>>>();
    cudaDeviceSynchronize();
}


