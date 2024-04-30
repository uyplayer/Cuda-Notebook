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
        sumMatrixOnGPU-2D-grid-1D-block.cu
        sumMatrixOnGPU_2D_grid_2D_block.cu
        sumMatrixOnGPU.cu
)

include_directories(../../../../common)

add_executable(chapter02 ${source_files})