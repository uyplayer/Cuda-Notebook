cmake_minimum_required(VERSION 3.18)

# set project
project(Cuda-Notebook LANGUAGES C CXX CUDA)

# cuda architecture
set(CMAKE_CUDA_ARCHITECTURES 86)


# config vcpkg ,put vcpkg dir into  system variable environment
set(VCPKG_ROOT "$ENV{vcpkg_root_dir}")
# vcpkg toolchain
SET(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
# vcpkg installed library header file location
include_directories("${VCPKG_ROOT}/installed/x64-windows/include")
message("vcpkg_DIR: ${VCPKG_ROOT}")
message("vcpkg.cmake dir: ${CMAKE_TOOLCHAIN_FILE}")
message("vcpkg header: ${VCPKG_ROOT}/installed/x64-windows/include")


# add common/include/lib/third_party
include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/lib)
link_directories(${PROJECT_SOURCE_DIR}/third_party)



add_subdirectory(third_party)
add_subdirectory(tests)
