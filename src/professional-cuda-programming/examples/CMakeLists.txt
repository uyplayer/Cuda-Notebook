






set(
        source_files
        main.cu
        checkDeviceInfor.cu
        checkDimension.cu
        checkThreadIndex.cu


)


set(
        common_files
        ../../../common/error.h
        ../../../common/GpuTimer.h

)


add_executable(examples ${source_files} ${common_files})