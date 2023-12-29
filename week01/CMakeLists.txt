

add_executable(week01  vectorAdd.cu
        main.cu)



target_include_directories(week01 PRIVATE ${CMAKE_SOURCE_DIR}/common)