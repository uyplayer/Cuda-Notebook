//
// Created by uyplayer on 2024-06-16.
//

#include <iostream>
#include <cuda_runtime.h>
#include "error.h"
#include "cxxopts.hpp"



#define NUM_BLOCKS 64
#define NUM_THREADS 256

int main(){
    std::cout << "Hello, clock!" << std::endl;
    cxxopts::Options options("asyncAPI", "A brief description");
    options.add_options()
            ("d,device", "Device ID", cxxopts::value<int>()->default_value("0"))
            ("t,thrust", "Using thrust", cxxopts::value<int>()->default_value("0"));
    const auto command_result = options.parse(argc, argv);
    int device_id = command_result["device"].as<int>();
    return 0;
}