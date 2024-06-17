




set(
        SRC
        main.cu
)

add_executable(
        cppOverload
        ${SRC}
)
target_include_directories(cppOverload PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(cppOverload PRIVATE cxxopts::cxxopts)