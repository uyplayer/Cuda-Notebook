cmake_minimum_required(VERSION 3.18)

project(Cuda-Notebook LANGUAGES C CXX CUDA)

# cuda architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

# Locate GTest
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
message("CMAKE_CURRENT_SOURCE_DIR:" ${CMAKE_CURRENT_SOURCE_DIR})



include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/libs)

include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# add sub dirs
add_subdirectory(tests)
add_subdirectory(check_configuration)
add_subdirectory(week01)
