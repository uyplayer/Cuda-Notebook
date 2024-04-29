set(
        source_files
        main.cu
        checkDeviceInfor.cu
        checkDimension.cu
        checkThreadIndex.cu
        defineGridBlock.cu
        sumArraysOnGPU-small-case.cu
        sumArraysOnGPU_timer.cu
        sumArraysOnHost.cu
        sumMatrixOnGPU_1D_grid_1D_block.cu
)

include_directories(${PROJECT_SOURCE_DIR}/common)

add_executable(examples ${source_files})