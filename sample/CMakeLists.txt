include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${Platform_SOURCE_DIR}/src/common
    ${Platform_SOURCE_DIR}/src/main
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/configured_files/include
    ${PyClassifiers_INCLUDE_DIRS}
    ${bayesnet_INCLUDE_DIRS}
)
add_executable(PlatformSample sample.cpp ${Platform_SOURCE_DIR}/src/main/Models.cpp) 
target_link_libraries(PlatformSample "${PyClassifiers}" "${BayesNet}" fimdlp ${Python3_LIBRARIES} "${TORCH_LIBRARIES}" ${LIBTORCH_PYTHON} ${Boost_LIBRARIES})