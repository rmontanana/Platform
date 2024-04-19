if(ENABLE_TESTING)
    set(TEST_PLATFORM "unit_tests_platform")
    include_directories(
        ${Platform_SOURCE_DIR}/src
        ${Platform_SOURCE_DIR}/lib/argparse/include
        ${Platform_SOURCE_DIR}/lib/mdlp
        ${Platform_SOURCE_DIR}/lib/Files
        ${Platform_SOURCE_DIR}/lib/json/include
        ${Platform_SOURCE_DIR}/lib/folding
        ${CMAKE_BINARY_DIR}/configured_files/include
        ${PyClassifiers_INCLUDE_DIRS}
        ${Bayesnet_INCLUDE_DIRS}
    )
    set(TEST_SOURCES_PLATFORM TestUtils.cpp TestPlatform.cpp TestResult.cpp ${Platform_SOURCE_DIR}/src/common/Datasets.cpp ${Platform_SOURCE_DIR}/src/common/Dataset.cpp)
    add_executable(${TEST_PLATFORM} ${TEST_SOURCES_PLATFORM})
    target_link_libraries(${TEST_PLATFORM} PUBLIC "${TORCH_LIBRARIES}" ArffFiles mdlp Catch2::Catch2WithMain BayesNet)
    add_test(NAME ${TEST_PLATFORM} COMMAND ${TEST_PLATFORM})
endif(ENABLE_TESTING)
