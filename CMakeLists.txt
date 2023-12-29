cmake_minimum_required(VERSION 3.18)

project(Cuda-Notebook LANGUAGES C CXX CUDA)

# cuda architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

# common
include_directories(${PROJECT_SOURCE_DIR}/common)

# include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}) equal to  code in blow 2 line
# this 2 ine only refer header
include_directories(${PROJECT_SOURCE_DIR}/lib/googletest-1.14.0/googlemock/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/googletest-1.14.0/googletest/include)
# also need add googletest's lib
set(GTEST_DIR "${PROJECT_SOURCE_DIR}/lib/googletest-1.14.0")
add_subdirectory(${GTEST_DIR})


# add sub dir
add_subdirectory(check_configuration)
add_subdirectory(tests)
add_subdirectory(week01)


