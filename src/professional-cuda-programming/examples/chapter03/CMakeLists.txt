







set(
        source_files
        main.cu

)

include_directories(${CMAKE_FIND_ROOT_PATH}/common)

add_executable(chapter03 ${source_files})