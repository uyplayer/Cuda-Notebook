




include_directories(../../../common)


find_package(OpenCV REQUIRED)
find_package(cJSON)

set(source_files
        main.cpp
        image_processing.cuh
        image_blur.cu
)



add_executable(image_processing ${source_files})

target_link_libraries(image_processing  cjson::cjson)
