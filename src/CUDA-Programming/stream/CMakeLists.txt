












set(
        source_files
        main.cu
        multi_stream.cu


        ../../../common/error.h
        ../../../common/GpuTimer.h
)


add_executable(stream ${source_files})