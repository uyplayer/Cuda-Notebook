cmake_minimum_required(VERSION 3.18)
project(Cuda-Notebook LANGUAGES C CXX CUDA)

# load GTest lib
set(GTest_DIR "${VCPKG_ROOT}\\installed\\x64-windows\\share\\gtest")
enable_testing()
find_package(GTest CONFIG REQUIRED)
# check
if (GTest_FOUND)
    message(STATUS "Google Test found: ${GTest_DIR}")
else ()
    message(FATAL_ERROR "Google Test not found")
endif ()

set(SOURCE
        ../src/week01/add1dim.cu
        ../src/week01/add2dim.cu
        ../src/week01/add3dim.cu
)

add_executable(week01_tests
        week01_tests.cu
        ${SOURCE}
)

include_directories(${PROJECT_SOURCE_DIR}/src/week01)

target_link_libraries(week01_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME week01_tests COMMAND week01_tests)
