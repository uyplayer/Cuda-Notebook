


#find_package(Torch  REQUIRED)
#if (Torch_FOUND)
#    message(STATUS "Torch library found")
#else ()
#    message(FATAL_ERROR "Torch library not found")
#endif ()
#message(STATUS "Torch include dir: ${TORCH_INCLUDE_DIRS}")

include(FetchContent)
FetchContent_Declare(
        rapidcsv
        GIT_REPOSITORY "https://github.com/d99kris/rapidcsv.git"
        GIT_TAG        "v8.50"
)
FetchContent_MakeAvailable(rapidcsv)

set(source_files
        main.cu
        linear_regression.cu
        data_loader.cpp
)

add_executable(neural_network ${source_files})
target_include_directories(neural_network PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(neural_network PRIVATE rapidcsv)

