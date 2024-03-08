include_directories(
    ${Platform_SOURCE_DIR}/src/common
    ${Platform_SOURCE_DIR}/src/main
    ${Python3_INCLUDE_DIRS}
    ${Platform_SOURCE_DIR}/lib/Files
    ${Platform_SOURCE_DIR}/lib/mdlp
    ${Platform_SOURCE_DIR}/lib/argparse/include
    ${Platform_SOURCE_DIR}/lib/folding
    ${Platform_SOURCE_DIR}/lib/json/include
    ${CMAKE_BINARY_DIR}/configured_files/include
    /usr/local/include
)
add_executable(PlatformSample sample.cc ${Platform_SOURCE_DIR}/src/main/Models.cc) 
target_link_libraries(PlatformSample "${PyClassifiers}" "${BayesNet}" ArffFiles mdlp ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} Boost::python Boost::numpy)