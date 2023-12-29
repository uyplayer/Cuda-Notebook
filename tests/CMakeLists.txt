

set(SOURCE_FILES
        test_week01.cu
        ../week01/vectorAdd.cu
)




add_executable(vector_add_test ${SOURCE_FILES})

target_link_libraries(vector_add_test  gtest   gtest_main)

target_include_directories(vector_add_test PRIVATE
        ../week01
)





