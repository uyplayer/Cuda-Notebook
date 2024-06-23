
set(
        SRC
        main.cu
        mergeSortHost.cpp
        mergeSortHost.cpp
)

add_executable(
        mergeSort
        ${SRC}
)
target_include_directories(mergeSort PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(mergeSort PRIVATE cxxopts::cxxopts)