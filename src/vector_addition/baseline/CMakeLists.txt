


set(
        SOURCE_FILES
        vectorAdd.cu
        ../../../common/GpuTimer.h
        ../../../common/error.h
)




add_executable(vectorAdd ${SOURCE_FILES})