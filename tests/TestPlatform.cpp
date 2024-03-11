#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <map>
#include <string>
#include "TestUtils.h"
#include "config.h"


TEST_CASE("Test Python Classifiers score", "[PyClassifiers]")
{
    std::string version = { platform_project_version.begin(), platform_project_version.end() };
    REQUIRE(version == "1.0.4");
}