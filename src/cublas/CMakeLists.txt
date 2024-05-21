

set(source_files
        main.cu
        based_indexing.cu
        handle_ops.cu

)

add_executable(cublas ${source_files})
target_include_directories(neural_network PRIVATE ../../common)

