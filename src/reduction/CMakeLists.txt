








set(source_files
        main.cu
        reduction_v1.cu
        reduction_v2.cu

)

add_executable(reduction ${source_files})
target_include_directories(reduction PRIVATE ../../common)

