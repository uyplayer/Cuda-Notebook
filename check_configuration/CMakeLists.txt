

add_executable(check_configuration
                check_config.cu
)

target_include_directories(check_configuration PRIVATE ${CMAKE_SOURCE_DIR}/common)