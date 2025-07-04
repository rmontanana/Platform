if(ENABLE_TESTING)
    set(TEST_PLATFORM "unit_tests_platform")
    include_directories(
        ${Platform_SOURCE_DIR}/src
        ${CMAKE_BINARY_DIR}/configured_files/include
    )
    set(TEST_SOURCES_PLATFORM 
        TestUtils.cpp TestPlatform.cpp TestResult.cpp TestScores.cpp TestDecisionTree.cpp TestAdaBoost.cpp
        ${Platform_SOURCE_DIR}/src/common/Datasets.cpp ${Platform_SOURCE_DIR}/src/common/Dataset.cpp ${Platform_SOURCE_DIR}/src/common/Discretization.cpp
        ${Platform_SOURCE_DIR}/src/main/Scores.cpp 
        ${Platform_SOURCE_DIR}/src/experimental_clfs/DecisionTree.cpp
        ${Platform_SOURCE_DIR}/src/experimental_clfs/AdaBoost.cpp
    )
    add_executable(${TEST_PLATFORM} ${TEST_SOURCES_PLATFORM})
    target_link_libraries(${TEST_PLATFORM} PUBLIC 
      torch::torch fimdlp:fimdlp Catch2::Catch2WithMain bayesnet::bayesnet pyclassifiers::pyclassifiers)
    add_test(NAME ${TEST_PLATFORM} COMMAND ${TEST_PLATFORM})
endif(ENABLE_TESTING)
