

set(
        SRC
        main.cu

)



add_executable(
        c++11_cuda
        ${SRC}
)
target_include_directories(c++11_cuda PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(c++11_cuda PRIVATE cxxopts::cxxopts)
