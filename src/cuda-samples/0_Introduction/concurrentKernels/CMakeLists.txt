













set(
    SRC
    main.cu
)

add_executable(
        concurrentKernels
        ${SRC}
)
target_include_directories(concurrentKernels PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(concurrentKernels PRIVATE cxxopts::cxxopts)