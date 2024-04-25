


set(
        SOURCE_FILES
        vectorAdd_um.cu
        ../../../common/GpuTimer.h
        ../../../common/error.h
)




add_executable(vectorAdd_um ${SOURCE_FILES})