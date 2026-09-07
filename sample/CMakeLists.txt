include_directories(
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/configured_files/include
    ${Platform_SOURCE_DIR}/src
    ${Platform_SOURCE_DIR}/src/common
    ${Platform_SOURCE_DIR}/src/main
)

# modelRegister.h instantiates every classifier, so the experimental ones have
# to be linked in as well.
add_executable(PlatformSample sample.cpp
    ${Platform_SOURCE_DIR}/src/main/Models.cpp
    ${Platform_SOURCE_DIR}/src/experimental_clfs/DecisionTree.cpp
    ${Platform_SOURCE_DIR}/src/experimental_clfs/AdaBoost.cpp
)
target_link_libraries(PlatformSample Boost::boost pyclassifiers::pyclassifiers bayesnet::bayesnet argparse::argparse fimdlp::fimdlp arff-files::arff-files folding::folding torch::torch)
