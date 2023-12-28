

add_executable(check_configuration
        check.cu
)

target_include_directories(check_configuration PRIVATE ${CMAKE_SOURCE_DIR}/common)