

set(source_files
        main.cu
        conv_cudnn.cu
)


# 设置CUDA和cuDNN路径
set(CUDNN_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDNN_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudnn.lib")

add_executable(cudnn ${source_files})
# 添加头文件路径
target_include_directories(cudnn PRIVATE ../../common ${CUDNN_INCLUDE_DIR})
# 添加链接库
target_link_libraries(cudnn PRIVATE ${CUDNN_LIBRARY})



