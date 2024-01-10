#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <map>
#include <string>
#include "TestUtils.h"

TEST_CASE("Test Python Classifiers score", "[PyClassifiers]")
{
    map <pair<std::string, std::string>, float> scores = {
        // Diabetes
        {{"diabetes", "STree"}, 0.81641}, {{"diabetes", "ODTE"}, 0.84635}, {{"diabetes", "SVC"}, 0.76823}, {{"diabetes", "RandomForest"}, 1.0},
        // Ecoli
        {{"ecoli", "STree"}, 0.8125}, {{"ecoli", "ODTE"}, 0.84821}, {{"ecoli", "SVC"}, 0.89583}, {{"ecoli", "RandomForest"}, 1.0},
        // Glass
        {{"glass", "STree"}, 0.57009}, {{"glass", "ODTE"}, 0.77103}, {{"glass", "SVC"}, 0.35514}, {{"glass", "RandomForest"}, 1.0},
        // Iris
        {{"iris", "STree"}, 0.99333}, {{"iris", "ODTE"}, 0.98667}, {{"iris", "SVC"}, 0.97333}, {{"iris", "RandomForest"}, 1.0},
    };

    std::string file_name = GENERATE("glass", "iris", "ecoli", "diabetes");
    auto raw = RawDatasets(file_name, false);

    SECTION("Test STree classifier (" + file_name + ")")
    {
        auto clf = pywrap::STree();
        clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
        auto score = clf.score(raw.Xt, raw.yt);
        REQUIRE(score == Catch::Approx(scores[{file_name, "STree"}]).epsilon(raw.epsilon));
    }
    SECTION("Test ODTE classifier (" + file_name + ")")
    {
        auto clf = pywrap::ODTE();
        clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
        auto score = clf.score(raw.Xt, raw.yt);
        REQUIRE(score == Catch::Approx(scores[{file_name, "ODTE"}]).epsilon(raw.epsilon));
    }
    SECTION("Test SVC classifier (" + file_name + ")")
    {
        auto clf = pywrap::SVC();
        clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
        auto score = clf.score(raw.Xt, raw.yt);
        REQUIRE(score == Catch::Approx(scores[{file_name, "SVC"}]).epsilon(raw.epsilon));
    }
    SECTION("Test RandomForest classifier (" + file_name + ")")
    {
        auto clf = pywrap::RandomForest();
        clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
        auto score = clf.score(raw.Xt, raw.yt);
        REQUIRE(score == Catch::Approx(scores[{file_name, "RandomForest"}]).epsilon(raw.epsilon));
    }
}
TEST_CASE("Classifiers features", "[PyClassifiers]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = pywrap::STree();
    clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
    REQUIRE(clf.getNumberOfNodes() == 3);
    REQUIRE(clf.getNumberOfEdges() == 2);
}
TEST_CASE("Get num features & num edges", "[PyClassifiers]")
{
    auto raw = RawDatasets("iris", true);
    auto clf = pywrap::ODTE();
    clf.fit(raw.Xt, raw.yt, raw.featurest, raw.classNamet, raw.statest);
    REQUIRE(clf.getNumberOfNodes() == 10);
    REQUIRE(clf.getNumberOfEdges() == 10);
}