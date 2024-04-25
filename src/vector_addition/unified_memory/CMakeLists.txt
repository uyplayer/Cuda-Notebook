


set(
        SOURCE_FILES
        vectorAdd_um_prefetch.cu
#        vectorAdd_um_baseline.cu
        ../../../common/GpuTimer.h
        ../../../common/error.h
)




add_executable(vectorAdd_um ${SOURCE_FILES})