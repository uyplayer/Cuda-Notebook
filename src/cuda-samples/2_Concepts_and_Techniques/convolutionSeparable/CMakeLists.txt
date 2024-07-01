











set(
    Source
    main.cu
)



add_executable(convolutionSeparable ${Source})

target_include_directories(convolutionSeparable PRIVATE ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(convolutionSeparable PRIVATE cxxopts::cxxopts ${OpenCV_LIBS})