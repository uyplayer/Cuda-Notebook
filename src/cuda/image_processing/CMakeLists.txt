

find_package(cJSON CONFIG REQUIRED)

find_package(cJSON)

set(source_files
        main.cpp
        image_processing.cuh
        image_blur.cu
)

add_executable(image_processing ${source_files})
target_include_directories(image_processing PRIVATE ../../../common)
target_link_libraries(image_processing PRIVATE cjson)
