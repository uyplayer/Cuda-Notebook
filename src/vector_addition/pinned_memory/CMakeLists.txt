


set(SOURCE_FILES
        vectorAdd_pinned.cu
        ../../../common/GpuTimer.h
        ../../../common/error.h
)




add_executable(pinned ${SOURCE_FILES})