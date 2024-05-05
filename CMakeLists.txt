cmake_minimum_required(VERSION 3.18)


# Set project
project(Cuda-Notebook LANGUAGES C CXX CUDA)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe")

# test conan package manager
#find_package(cJSON)
#add_executable(Cuda-Notebook main.cpp)
#target_link_libraries(Cuda-Notebook cjson::cjson)



add_subdirectory(src/cuda/image_processing)