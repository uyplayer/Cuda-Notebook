cmake_minimum_required(VERSION 3.18)

# Set project
project(Cuda-Notebook LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)


if(DEFINED ENV{vcpkg_root_dir})
    set(VCPKG_ROOT_DIR $ENV{vcpkg_root_dir})
else()
    message(FATAL_ERROR "Environment variable 'vcpkg_root_dir' is not set.")
endif()

message(STATUS "VCPKG_ROOT_DIR: ${VCPKG_ROOT_DIR}")

set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT_DIR}/scripts/buildsystems/vcpkg.cmake")

find_package(CUDA REQUIRED)
enable_language(CUDA)

message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_FOUND: ${OpenCV_FOUND}")
include_directories(${CUDA_INCLUDE_DIRS})






include_directories(${PROJECT_SOURCE_DIR}/common)

#find_package(cJSON CONFIG REQUIRED)
#add_executable(main main.cpp)
#target_link_libraries(main PRIVATE cjson)

add_subdirectory(src/cuda/image_processing)
add_subdirectory(src/cuda/neural_network)
