
set(
        SRC
        main.cu
        boxFilter.h
        boxFilter.cuh

)

find_package(OpenCV CONFIG REQUIRED)
add_executable(
        boxFilter
        ${SRC}
)
target_include_directories(boxFilter PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(boxFilter PRIVATE cxxopts::cxxopts ${OpenCV_LIBS})
