// Platform#8 / Platform#9: b_main and b_grid experiment build the result of a dataset with the
// same functions (main/FoldRun.h). These tests pin the pure part: the seed x fold order, the
// confusion matrices and the fields of PartialResult.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "main/FoldRun.h"
#include "main/Scores.h"

using json = nlohmann::ordered_json;

namespace {
    const std::vector<int> kSeeds = { 42, 7, 1024 }; // deliberately not sorted: b_main keeps the given order
    const int kFolds = 2;
    const std::vector<std::string> kLabels = { "a", "b", "c" };

    platform::FoldOutcome fold(int seed, int n_fold)
    {
        platform::FoldOutcome f;
        f.seed = seed;
        f.fold = n_fold;
        f.score_test = 0.5 + 0.01 * n_fold + seed / 10000.0;
        f.score_train = 0.9;
        f.has_train = true;
        f.time_train = 1.0;
        f.time_test = 0.1;
        f.nodes = 10;
        f.edges = 20;
        f.states = 30;
        // diagonal = seed, so each matrix identifies its fold
        f.cm_test = { seed, 1, 0, 0, n_fold, 0, 2, 0, 3 };
        f.cm_train = f.cm_test;
        f.notes = { "note " + std::to_string(seed) + "/" + std::to_string(n_fold) };
        f.graph = "digraph " + std::to_string(seed);
        return f;
    }
    std::vector<platform::FoldOutcome> shuffled()
    {
        std::vector<platform::FoldOutcome> folds;
        for (int seed : kSeeds)
            for (int n = 0; n < kFolds; ++n)
                folds.push_back(fold(seed, n));
        std::mt19937 g(3);
        std::shuffle(folds.begin(), folds.end(), g);
        return folds;
    }
    platform::FoldOptions options(bool discretized, bool no_train_score)
    {
        platform::FoldOptions o;
        o.discretized = discretized;
        o.no_train_score = no_train_score;
        o.graph = true;
        o.num_classes = 3;
        o.labels = kLabels;
        return o;
    }
}

TEST_CASE("Folds in seed x fold order", "[FoldResult]")
{
    auto folds = shuffled();
    platform::sort_by_seed_fold(folds, kSeeds, kFolds);
    REQUIRE(folds.size() == kSeeds.size() * kFolds);
    for (size_t i = 0; i < folds.size(); ++i) {
        REQUIRE(folds[i].seed == kSeeds[i / kFolds]);
        REQUIRE(folds[i].fold == static_cast<int>(i % kFolds));
    }
}

TEST_CASE("Incomplete or duplicated folds are rejected", "[FoldResult]")
{
    auto folds = shuffled();
    SECTION("missing")
    {
        folds.pop_back();
        REQUIRE_THROWS_AS(platform::sort_by_seed_fold(folds, kSeeds, kFolds), std::runtime_error);
    }
    SECTION("duplicated")
    {
        folds.back() = folds.front();
        REQUIRE_THROWS_AS(platform::sort_by_seed_fold(folds, kSeeds, kFolds), std::runtime_error);
    }
    SECTION("unknown seed")
    {
        folds.back().seed = 99;
        REQUIRE_THROWS_AS(platform::sort_by_seed_fold(folds, kSeeds, kFolds), std::runtime_error);
    }
}

TEST_CASE("Confusion matrix JSON as Scores", "[FoldResult]")
{
    // Multiclass with a class absent from the test partition (class 2 never occurs)
    std::vector<int> actual = { 0, 0, 1, 1, 1, 0, 1, 0 };
    std::vector<int> predicted = { 0, 1, 1, 2, 1, 0, 0, 2 };
    auto y = torch::tensor(actual, torch::kInt32);
    auto proba = torch::zeros({ static_cast<long>(actual.size()), 3 }, torch::kFloat32);
    for (size_t i = 0; i < predicted.size(); ++i)
        proba[i][predicted[i]] = 1.0;
    platform::Scores scores(y, proba, 3, kLabels);
    // Vectorized confusion matrix (bincount) against a plain count
    std::vector<int> expected(9, 0);
    for (size_t i = 0; i < actual.size(); ++i)
        expected[actual[i] * 3 + predicted[i]]++;
    auto cm = scores.get_confusion_matrix().contiguous();
    REQUIRE(cm.dtype() == torch::kInt32);
    REQUIRE(std::vector<int>(cm.data_ptr<int>(), cm.data_ptr<int>() + 9) == expected);
    REQUIRE(platform::confusion_matrix_json(expected, 3, kLabels) == scores.get_confusion_matrix_json(true));
    REQUIRE_THROWS_AS(platform::confusion_matrix_json({ 1, 2, 3 }, 3, kLabels), std::runtime_error);
}

TEST_CASE("Result of a dataset from its folds", "[FoldResult]")
{
    auto folds = shuffled();
    platform::sort_by_seed_fold(folds, kSeeds, kFolds);
    SECTION("discretized, with train score")
    {
        auto j = platform::build_partial_result(folds, options(true, false), "ds", 100, 4, 3, json::object()).getJson();
        REQUIRE(j["scores_test"].size() == folds.size());
        REQUIRE(j["scores_test"][0].get<double>() == Catch::Approx(folds[0].score_test));
        REQUIRE(j["confusion_matrices"].size() == folds.size());
        REQUIRE(j["confusion_matrices"][0]["a"][0].get<int>() == kSeeds[0]);
        REQUIRE(j["confusion_matrices"][2]["a"][0].get<int>() == kSeeds[1]);
        REQUIRE(j["confusion_matrices_train"].size() == folds.size());
        REQUIRE(j["notes"][0].get<std::string>() == "Seed: 42 Fold: 0: note 42/0");
        REQUIRE(j["notes"][3].get<std::string>() == "Seed: 7 Fold: 1: note 7/1");
        REQUIRE(j["graph"].size() == folds.size());
        REQUIRE(j["score_train"].get<double>() == Catch::Approx(0.9));
        REQUIRE(j["nodes"].get<double>() == Catch::Approx(10));
    }
    SECTION("not discretized: no matrices, as b_main")
    {
        auto j = platform::build_partial_result(folds, options(false, false), "ds", 100, 4, 3, json::object()).getJson();
        REQUIRE(j["confusion_matrices"].empty());
        REQUIRE(j["confusion_matrices_train"].empty());
    }
    SECTION("--no-train-score: no train matrices key, as b_main")
    {
        for (auto& f : folds) {
            f.score_train = 0.0;
            f.has_train = false;
            f.cm_train.clear();
        }
        auto j = platform::build_partial_result(folds, options(true, true), "ds", 100, 4, 3, json::object()).getJson();
        REQUIRE_FALSE(j.contains("confusion_matrices_train"));
        REQUIRE(j["score_train"].get<double>() == 0.0);
    }
}
