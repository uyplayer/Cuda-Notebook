

set(source_files
        main.cu
        0-based.cu
        1-based.cu

)

add_executable(cublas ${source_files})
target_include_directories(neural_network PRIVATE ../../common)

