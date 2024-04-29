





set(
        source_files
        main.cu

        ../../../common/GpuTimer.h
        ../../../common/error.h
)



add_executable(warp ${source_files})