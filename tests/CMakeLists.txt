cmake_minimum_required(VERSION 3.18)
project(Cuda-Notebook LANGUAGES C CXX CUDA)


# Check if Catch2 was found
find_package(Catch2  REQUIRED)

if (Catch2_FOUND)
    message(STATUS "Catch2 found: ${Catch2_DIR}")
else ()
    message(FATAL_ERROR "Catch2 not found")
endif ()


set(SOURCE
        ../src/week01/add1dim.cu
        ../src/week01/add2dim.cu
        ../src/week01/add3dim.cu
        ../src/week01/transpose.cu
        ../src/week01/shared.cu
        ../src/week01/const.cu
)

add_executable(week01_tests
        week01_tests.cu
        ${SOURCE}
)

# Include Doctest's header files
include_directories(${doctest_INCLUDE_DIRS})

target_link_libraries(week01_tests PRIVATE
        Catch2::Catch2WithMain
)
