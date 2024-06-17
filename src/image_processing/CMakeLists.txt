

find_package(OpenCV CONFIG REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

set(source_files
        main.cpp
        image_processing.cuh
        Base.cpp
        image_blur.cu
        cuda_kernel.cu

)

add_executable(image_processing ${source_files})
target_include_directories(image_processing PRIVATE ../../common)
target_link_libraries(image_processing PRIVATE  ${OpenCV_LIBS})
