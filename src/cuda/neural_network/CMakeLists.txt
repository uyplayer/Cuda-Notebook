



find_package(OpenCV  REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

set(source_files
        main.cpp
)

add_executable(neural_network ${source_files})
target_include_directories(neural_network PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(neural_network PRIVATE  ${OpenCV_LIBS})
