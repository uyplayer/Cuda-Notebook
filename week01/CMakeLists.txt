# 查找 Google Test 库
find_package(GTest REQUIRED)

# 设置项目源文件
set(SOURCES
        main.cu
        vectorAdd.cu
        add2dim.cu
        add3dim.cu
        week01.h
)

# 创建主项目目标
add_executable(week01 ${SOURCES})

# 包括项目头文件路径
target_include_directories(week01 PRIVATE ${CMAKE_SOURCE_DIR}/common)




