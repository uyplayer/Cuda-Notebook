


set(SOURCES
        block_clusters.cu
        main.cu
)

# 创建主项目目标
add_executable(week02 ${SOURCES})

# 包括项目头文件路径
target_include_directories(week02 PRIVATE ${CMAKE_SOURCE_DIR}/common)