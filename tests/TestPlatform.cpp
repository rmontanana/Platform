#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <map>
#include <string>
#include "TestUtils.h"
#include "folding.hpp"
#include <bayesnet/classifiers/TAN.h>
#include "config.h"


TEST_CASE("Test Platform version", "[Platform]")
{
    std::string version = { platform_project_version.begin(), platform_project_version.end() };
    REQUIRE(version == "1.0.4");
}
TEST_CASE("Test Folding library version", "[Folding]")
{
    auto kfold = folding::KFold(5, 100);
    REQUIRE(kfold.version() == "1.0.1");
}
TEST_CASE("Test BayesNet version", "[BayesNet]")
{
    std::string version = bayesnet::TAN().getVersion();
    REQUIRE(version == "1.0.4");
}
TEST_CASE("Test mdlp version", "[mdlp]")
{
    std::string version = mdlp::CPPFImdlp::version();
    REQUIRE(version == "1.1.2");
}