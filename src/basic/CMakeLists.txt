






set(SOURCE_FILES
        main.cu
        add_kernel.cu
        ../../common/GpuTimer.h
        ../../common/error.h
        add_vector.cu

)

add_executable(basic ${SOURCE_FILES})