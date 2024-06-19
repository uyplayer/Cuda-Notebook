




set(
        SRC
        main.cu
)


add_executable(
        asyncAPI
        ${SRC}
)
target_include_directories(asyncAPI PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(asyncAPI PRIVATE cxxopts::cxxopts)
