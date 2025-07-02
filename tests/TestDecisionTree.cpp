// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <torch/torch.h>
#include <memory>
#include <stdexcept>
#include "experimental_clfs/DecisionTree.h"
#include "TestUtils.h"

using namespace bayesnet;
using namespace Catch::Matchers;

TEST_CASE("DecisionTree Construction", "[DecisionTree]")
{
    SECTION("Default constructor")
    {
        REQUIRE_NOTHROW(DecisionTree());
    }

    SECTION("Constructor with parameters")
    {
        REQUIRE_NOTHROW(DecisionTree(5, 10, 3));
    }
}

TEST_CASE("DecisionTree Hyperparameter Setting", "[DecisionTree]")
{
    DecisionTree dt;

    SECTION("Set individual hyperparameters")
    {
        REQUIRE_NOTHROW(dt.setMaxDepth(10));
        REQUIRE_NOTHROW(dt.setMinSamplesSplit(5));
        REQUIRE_NOTHROW(dt.setMinSamplesLeaf(2));
        REQUIRE(dt.getMaxDepth() == 10);
        REQUIRE(dt.getMinSamplesSplit() == 5);
        REQUIRE(dt.getMinSamplesLeaf() == 2);
    }

    SECTION("Set hyperparameters via JSON")
    {
        nlohmann::json params;
        params["max_depth"] = 7;
        params["min_samples_split"] = 4;
        params["min_samples_leaf"] = 2;

        REQUIRE_NOTHROW(dt.setHyperparameters(params));
        REQUIRE(dt.getMaxDepth() == 7);
        REQUIRE(dt.getMinSamplesSplit() == 4);
        REQUIRE(dt.getMinSamplesLeaf() == 2);
    }

    SECTION("Invalid hyperparameters should throw")
    {
        nlohmann::json params;

        // Negative max_depth
        params["max_depth"] = -1;
        REQUIRE_THROWS_AS(dt.setHyperparameters(params), std::invalid_argument);

        // Zero min_samples_split
        params["max_depth"] = 5;
        params["min_samples_split"] = 0;
        REQUIRE_THROWS_AS(dt.setHyperparameters(params), std::invalid_argument);

        // Negative min_samples_leaf
        params["min_samples_split"] = 2;
        params["min_samples_leaf"] = -5;
        REQUIRE_THROWS_AS(dt.setHyperparameters(params), std::invalid_argument);
    }
}

TEST_CASE("DecisionTree Basic Functionality", "[DecisionTree]")
{
    // Create a simple dataset
    int n_samples = 20;
    int n_features = 2;

    std::vector<std::vector<int>> X(n_features, std::vector<int>(n_samples));
    std::vector<int> y(n_samples);

    // Simple pattern: class depends on first feature
    for (int i = 0; i < n_samples; i++) {
        X[0][i] = i < 10 ? 0 : 1;
        X[1][i] = i % 2;
        y[i] = X[0][i];  // Class equals first feature
    }

    std::vector<std::string> features = { "f1", "f2" };
    std::string className = "class";
    std::map<std::string, std::vector<int>> states;
    states["f1"] = { 0, 1 };
    states["f2"] = { 0, 1 };
    states["class"] = { 0, 1 };

    SECTION("Training with vector interface")
    {
        DecisionTree dt(3, 2, 1);
        REQUIRE_NOTHROW(dt.fit(X, y, features, className, states, Smoothing_t::NONE));

        auto predictions = dt.predict(X);
        REQUIRE(predictions.size() == static_cast<size_t>(n_samples));

        // Should achieve perfect accuracy on this simple dataset
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == y[i]) correct++;
        }
        REQUIRE(correct == n_samples);
    }

    SECTION("Prediction before fitting")
    {
        DecisionTree dt;
        REQUIRE_THROWS_WITH(dt.predict(X),
            ContainsSubstring("Classifier has not been fitted"));
    }

    SECTION("Probability predictions")
    {
        DecisionTree dt(3, 2, 1);
        dt.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto proba = dt.predict_proba(X);
        REQUIRE(proba.size() == static_cast<size_t>(n_samples));
        REQUIRE(proba[0].size() == 2);  // Two classes

        // Check probabilities sum to 1 and probabilities are valid
        auto predictions = dt.predict(X);
        for (size_t i = 0; i < proba.size(); i++) {
            auto p = proba[i];
            auto pred = predictions[i];
            REQUIRE(p.size() == 2);
            REQUIRE(p[0] >= 0.0);
            REQUIRE(p[1] >= 0.0);
            double sum = p[0] + p[1];
            //Check that prodict_proba matches the expected predict value
            REQUIRE(pred == (p[0] > p[1] ? 0 : 1));
            REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-6));
        }
    }
}

TEST_CASE("DecisionTree on Iris Dataset", "[DecisionTree][iris]")
{
    auto raw = RawDatasets("iris", true);

    SECTION("Training with dataset format")
    {
        DecisionTree dt(5, 2, 1);

        INFO("Dataset shape: " << raw.dataset.sizes());
        INFO("Features: " << raw.featurest.size());
        INFO("Samples: " << raw.nSamples);

        // DecisionTree expects dataset in format: features x samples, with labels as last row
        REQUIRE_NOTHROW(dt.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE));

        // Test prediction
        auto predictions = dt.predict(raw.Xt);
        REQUIRE(predictions.size(0) == raw.yt.size(0));

        // Calculate accuracy
        auto correct = torch::sum(predictions == raw.yt).item<int>();
        double accuracy = static_cast<double>(correct) / raw.yt.size(0);
        double acurracy_computed = dt.score(raw.Xt, raw.yt);
        REQUIRE(accuracy > 0.97);  // Reasonable accuracy for Iris
        REQUIRE(acurracy_computed == Catch::Approx(accuracy).epsilon(1e-6));
    }

    SECTION("Training with vector interface")
    {
        DecisionTree dt(5, 2, 1);

        REQUIRE_NOTHROW(dt.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv, Smoothing_t::NONE));

        // std::cout << "Tree structure:\n";
        // auto graph_lines = dt.graph("Iris Decision Tree");
        // for (const auto& line : graph_lines) {
        //     std::cout << line << "\n";
        // }
        auto predictions = dt.predict(raw.Xv);
        REQUIRE(predictions.size() == raw.yv.size());
    }

    SECTION("Different tree depths")
    {
        std::vector<int> depths = { 1, 3, 5 };

        for (int depth : depths) {
            DecisionTree dt(depth, 2, 1);
            dt.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

            auto predictions = dt.predict(raw.Xt);
            REQUIRE(predictions.size(0) == raw.yt.size(0));
        }
    }
}

TEST_CASE("DecisionTree Edge Cases", "[DecisionTree]")
{
    auto raw = RawDatasets("iris", true);

    SECTION("Very shallow tree")
    {
        DecisionTree dt(1, 2, 1);  // depth = 1
        dt.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

        auto predictions = dt.predict(raw.Xt);
        REQUIRE(predictions.size(0) == raw.yt.size(0));

        // With depth 1, should have at most 2 unique predictions
        auto unique_vals = at::_unique(predictions);
        REQUIRE(std::get<0>(unique_vals).size(0) <= 2);
    }

    SECTION("High min_samples_split")
    {
        DecisionTree dt(10, 50, 1);
        dt.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

        auto predictions = dt.predict(raw.Xt);
        REQUIRE(predictions.size(0) == raw.yt.size(0));
    }
}

TEST_CASE("DecisionTree Graph Visualization", "[DecisionTree]")
{
    // Simple dataset
    std::vector<std::vector<int>> X = { {0,0,0,1}, {0,1,1,1} };  // XOR pattern
    std::vector<int> y = { 0, 1, 1, 0 };  // XOR pattern
    std::vector<std::string> features = { "x1", "x2" };
    std::string className = "xor";
    std::map<std::string, std::vector<int>> states;
    states["x1"] = { 0, 1 };
    states["x2"] = { 0, 1 };
    states["xor"] = { 0, 1 };

    SECTION("Graph generation")
    {
        DecisionTree dt(2, 1, 1);
        dt.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto graph_lines = dt.graph();

        REQUIRE(graph_lines.size() > 2);
        REQUIRE(graph_lines.front() == "digraph DecisionTree {");
        REQUIRE(graph_lines.back() == "}");

        // Should contain node definitions
        bool has_nodes = false;
        for (const auto& line : graph_lines) {
            if (line.find("node") != std::string::npos) {
                has_nodes = true;
                break;
            }
        }
        REQUIRE(has_nodes);
    }

    SECTION("Graph with title")
    {
        DecisionTree dt(2, 1, 1);
        dt.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto graph_lines = dt.graph("XOR Tree");

        bool has_title = false;
        for (const auto& line : graph_lines) {
            if (line.find("label=\"XOR Tree\"") != std::string::npos) {
                has_title = true;
                break;
            }
        }
        REQUIRE(has_title);
    }
}

TEST_CASE("DecisionTree with Weights", "[DecisionTree]")
{
    auto raw = RawDatasets("iris", true);

    SECTION("Uniform weights")
    {
        DecisionTree dt(5, 2, 1);
        dt.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, raw.weights, Smoothing_t::NONE);

        auto predictions = dt.predict(raw.Xt);
        REQUIRE(predictions.size(0) == raw.yt.size(0));
    }

    SECTION("Non-uniform weights")
    {
        auto weights = torch::ones({ raw.nSamples });
        weights.index({ torch::indexing::Slice(0, 50) }) *= 2.0;  // Emphasize first class
        weights = weights / weights.sum();

        DecisionTree dt(5, 2, 1);
        dt.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, weights, Smoothing_t::NONE);

        auto predictions = dt.predict(raw.Xt);
        REQUIRE(predictions.size(0) == raw.yt.size(0));
    }
}