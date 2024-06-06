

set(source_files
        main.cu
        conv_cudnn.cu

)

add_executable(cublas ${source_files})
target_include_directories(neural_network PRIVATE ../../common)