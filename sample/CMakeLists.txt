include_directories(
    ${Platform_SOURCE_DIR}/src/Platform
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/src/PyClassifiers
    ${Python3_INCLUDE_DIRS}
    ${Platform_SOURCE_DIR}/lib/Files
    ${Platform_SOURCE_DIR}/lib/argparse/include
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/src/BayesNet
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/folding
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/mdlp
    ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/json/include
    ${CMAKE_BINARY_DIR}/configured_files/include
)
add_executable(PlatformSample sample.cc ${Platform_MODULES}/Models.cc) 
target_link_libraries(PlatformSample PyClassifiers ArffFiles mdlp "${TORCH_LIBRARIES}")