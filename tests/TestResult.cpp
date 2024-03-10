#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "Result.h"
#include <filesystem>

TEST_CASE("Result class tests", "[Result]")
{
    std::string testPath = "test_data";
    std::string testFile = "test.json";

    SECTION("Constructor and load method")
    {
        platform::Result result;
        result.load(testPath, testFile);
        REQUIRE(result.date != "");
        REQUIRE(result.score >= 0);
        REQUIRE(result.scoreName != "");
        REQUIRE(result.title != "");
        REQUIRE(result.duration >= 0);
        REQUIRE(result.model != "");
    }

    SECTION("to_string method")
    {
        platform::Result result(testPath, testFile);
        result.load();
        std::string resultStr = result.to_string(1);
        REQUIRE(resultStr != "");
    }

    SECTION("Exception handling in load method")
    {
        std::string invalidFile = "invalid.json";
        auto result = platform::Result();
        REQUIRE_THROWS_AS(platform::result.load(testPath, invalidFile), std::invalid_argument);
    }
}