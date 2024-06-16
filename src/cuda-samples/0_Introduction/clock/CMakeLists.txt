



set(
    SRC
    main.cu
)

add_executable(
        clock
        ${SRC}
)
target_include_directories(clock PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(clock PRIVATE cxxopts::cxxopts)