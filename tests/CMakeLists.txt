
set(SOURCE_FILES
        test_week01.cu
        ../week01/vectorAdd.cu
)

add_executable(vector_add_test ${SOURCE_FILES})


if(TARGET gtest_main)
    message(STATUS "Linking with gtest_main")
    target_link_libraries(vector_add_test gtest_main)
elseif(TARGET gtest)
    message(STATUS "Linking with gtest")
    target_link_libraries(vector_add_test gtest)
else()
    message(FATAL_ERROR "Google Test not found")
endif()



target_include_directories(vector_add_test PRIVATE
        ../week01
)


