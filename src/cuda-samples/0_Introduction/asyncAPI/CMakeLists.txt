




set(
        SRC
        asyncAPI.cu
)


add_executable(
        asyncAPI
        ${SRC}
)
target_include_directories(asyncAPI PRIVATE ${PROJECT_SOURCE_DIR}/common)

