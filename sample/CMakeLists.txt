include_directories(
    ${Platform_SOURCE_DIR}/src/Platform
    ${Platform_SOURCE_DIR}/src/PyClassifiers
    ${Python3_INCLUDE_DIRS}
    ${Platform_SOURCE_DIR}/lib/Files
    ${Platform_SOURCE_DIR}/lib/mdlp
    ${Platform_SOURCE_DIR}/lib/argparse/include
    ${Platform_SOURCE_DIR}/lib/json/include
)
add_executable(PlatformSample sample.cc ${Platform_SOURCE_DIR}/src/Platform/Models.cc) 
target_link_libraries(PlatformSample Platform ArffFiles mdlp "${TORCH_LIBRARIES}" PyWrap)