cmake_minimum_required(VERSION 3.18)

project(Cuda-Notebook LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)



include_directories(${PROJECT_SOURCE_DIR}/common)

#include_directories(D:/software/installed/cuda development/include)


# 为每个章节添加子目录
add_subdirectory(check_configuration)
add_subdirectory(week01)

# 假设你的可执行文件叫做 cuda_app
# target_link_libraries(cuda_app ${CUDA_LIBRARIES})
# target_include_directories(cuda_app PRIVATE ${CUDA_INCLUDE_DIRS})
