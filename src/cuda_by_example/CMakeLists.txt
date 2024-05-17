



message(STATUS "cuda_by_example: ${CMAKE_CURRENT_SOURCE_DIR}")



set(source_files
        main.cu
        julia_gpu.cu

)

add_executable(cuda_by_example ${source_files})
target_include_directories(cuda_by_example PRIVATE ${PROJECT_SOURCE_DIR}/common)

