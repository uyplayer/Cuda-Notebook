






message(STATUS "thrust: ${CMAKE_CURRENT_SOURCE_DIR}")



set(source_files
        main.cu
        seqCuda.cu

)

add_executable(thrust ${source_files})
target_include_directories(thrust PRIVATE ${PROJECT_SOURCE_DIR}/common)

