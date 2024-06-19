

set(
        SRC
        main.cu
)

add_executable(
        fp16ScalarProduct
        ${SRC}
)
target_include_directories(fp16ScalarProduct PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(fp16ScalarProduct PRIVATE cxxopts::cxxopts)