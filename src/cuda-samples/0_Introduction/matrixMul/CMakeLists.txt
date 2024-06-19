



set(
        SRC
        main.cu
)

add_executable(
        matrixMul
        ${SRC}
)
target_include_directories(matrixMul PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(matrixMul PRIVATE cxxopts::cxxopts)