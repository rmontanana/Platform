if(ENABLE_TESTING)
    set(TEST_PLATFORM "unit_tests_platform")
    include_directories(
        ${Platform_SOURCE_DIR}/src/Platform
        ${Platform_SOURCE_DIR}/lib/argparse/include
        ${Platform_SOURCE_DIR}/lib/PyClassifiers/lib/BayesNet/lib/mdlp
        ${Platform_SOURCE_DIR}/lib/Files
        ${CMAKE_BINARY_DIR}/configured_files/include
    )
    set(TEST_SOURCES_PLATFORM TestUtils.cc TestPlatform.cc)
    add_executable(${TEST_PLATFORM} ${TEST_SOURCES_PLATFORM})
    target_link_libraries(${TEST_PLATFORM} PUBLIC "${TORCH_LIBRARIES}" ArffFiles mdlp Catch2::Catch2WithMain)
    add_test(NAME ${TEST_PLATFORM} COMMAND ${TEST_PLATFORM})
endif(ENABLE_TESTING)