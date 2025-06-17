if(ENABLE_TESTING)
    set(TEST_PLATFORM "unit_tests_platform")
    include_directories(
        ${Platform_SOURCE_DIR}/src
        ${Platform_SOURCE_DIR}/lib/argparse/include
        ${Platform_SOURCE_DIR}/lib/mdlp/src
        ${Platform_SOURCE_DIR}/lib/Files
        ${Platform_SOURCE_DIR}/lib/json/include
        ${Platform_SOURCE_DIR}/lib/folding
        ${CMAKE_BINARY_DIR}/configured_files/include
        ${PyClassifiers_INCLUDE_DIRS}
        ${Bayesnet_INCLUDE_DIRS}
    )
    set(TEST_SOURCES_PLATFORM 
        TestUtils.cpp TestPlatform.cpp TestResult.cpp TestScores.cpp TestDecisionTree.cpp
        ${Platform_SOURCE_DIR}/src/common/Datasets.cpp ${Platform_SOURCE_DIR}/src/common/Dataset.cpp ${Platform_SOURCE_DIR}/src/common/Discretization.cpp
        ${Platform_SOURCE_DIR}/src/main/Scores.cpp ${Platform_SOURCE_DIR}/src/experimental_clfs/DecisionTree.cpp
    )
    add_executable(${TEST_PLATFORM} ${TEST_SOURCES_PLATFORM})
    target_link_libraries(${TEST_PLATFORM} PUBLIC "${TORCH_LIBRARIES}" fimdlp Catch2::Catch2WithMain bayesnet)
    add_test(NAME ${TEST_PLATFORM} COMMAND ${TEST_PLATFORM})
endif(ENABLE_TESTING)
