




set(source_files
        main.cu
        matmul_cpu.cu
        matmul_gpu.cu
        matmul.cuh

)

add_executable(matrix_mul ${source_files})
target_include_directories(matrix_mul PRIVATE ../../common)

