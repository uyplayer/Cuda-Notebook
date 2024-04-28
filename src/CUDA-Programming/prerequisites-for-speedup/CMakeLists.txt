




set(
        source_files
        main.cu
        add1cpu.cu
        add2gpu.cu
        ../../../common/error.h
        ../../../common/GpuTimer.h
)


add_executable(
        prerequisites-for-speedup
        ${source_files}
)