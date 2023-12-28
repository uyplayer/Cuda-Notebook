cmake_minimum_required(VERSION 3.18)

project(Cuda-Notebook LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)

# 包含头文件目录
include_directories(common)

## 包含库文件目录
#link_directories(${CMAKE_SOURCE_DIR}/lib)
#
## 为每个章节添加子目录
#add_subdirectory(chapter03)


