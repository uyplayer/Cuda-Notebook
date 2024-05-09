


#find_package(Torch  REQUIRED)
#if (Torch_FOUND)
#    message(STATUS "Torch library found")
#else ()
#    message(FATAL_ERROR "Torch library not found")
#endif ()
#message(STATUS "Torch include dir: ${TORCH_INCLUDE_DIRS}")

set(source_files
        main.cpp
        linear_regression.cu
)

add_executable(neural_network ${source_files})
target_include_directories(neural_network PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(neural_network PRIVATE  ${TORCH_LIBRARIES})
