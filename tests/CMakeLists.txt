if(ENABLE_TESTING)
    set(TEST_PLATFORM "unit_tests_platform")
    include_directories(
        ${BayesNet_SOURCE_DIR}/src/BayesNet
        ${BayesNet_SOURCE_DIR}/src/Platform
        ${BayesNet_SOURCE_DIR}/lib/Files
        ${BayesNet_SOURCE_DIR}/lib/mdlp
        ${BayesNet_SOURCE_DIR}/lib/json/include
        ${BayesNet_SOURCE_DIR}/lib/argparse/include
    )
    set(TEST_SOURCES_PLATFORM TestUtils.cc)
    add_executable(${TEST_PLATFORM} ${TEST_SOURCES_PLATFORM})
    target_link_libraries(${TEST_PLATFORM} PUBLIC "${TORCH_LIBRARIES}" ArffFiles mdlp Catch2::Catch2WithMain)
    add_test(NAME ${TEST_PLATFORM} COMMAND ${TEST_PLATFORM})
endif(ENABLE_TESTING)
