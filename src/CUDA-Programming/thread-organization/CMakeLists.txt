


set(
        source_files
        main.cu
        hello.cu

        ../../../common/error.h
        ../../../common/GpuTimer.h


)



add_executable(thread-organization ${source_files})