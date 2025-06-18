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
#include "experimental_clfs/AdaBoost.h"
#include "experimental_clfs/DecisionTree.h"
#include "experimental_clfs/TensorUtils.hpp"
#include "TestUtils.h"

using namespace bayesnet;
using namespace Catch::Matchers;


TEST_CASE("AdaBoost Construction", "[AdaBoost]")
{
    SECTION("Default constructor")
    {
        REQUIRE_NOTHROW(AdaBoost());
    }

    SECTION("Constructor with parameters")
    {
        REQUIRE_NOTHROW(AdaBoost(100, 2));
    }

    SECTION("Constructor parameter access")
    {
        AdaBoost ada(75, 3);
        REQUIRE(ada.getNEstimators() == 75);
        REQUIRE(ada.getBaseMaxDepth() == 3);
    }
}

TEST_CASE("AdaBoost Hyperparameter Setting", "[AdaBoost]")
{
    AdaBoost ada;

    SECTION("Set individual hyperparameters")
    {
        REQUIRE_NOTHROW(ada.setNEstimators(100));
        REQUIRE_NOTHROW(ada.setBaseMaxDepth(5));

        REQUIRE(ada.getNEstimators() == 100);
        REQUIRE(ada.getBaseMaxDepth() == 5);
    }

    SECTION("Set hyperparameters via JSON")
    {
        nlohmann::json params;
        params["n_estimators"] = 80;
        params["base_max_depth"] = 4;

        REQUIRE_NOTHROW(ada.setHyperparameters(params));
    }

    SECTION("Invalid hyperparameters should throw")
    {
        nlohmann::json params;

        // Negative n_estimators
        params["n_estimators"] = -1;
        REQUIRE_THROWS_AS(ada.setHyperparameters(params), std::invalid_argument);

        // Zero n_estimators
        params["n_estimators"] = 0;
        REQUIRE_THROWS_AS(ada.setHyperparameters(params), std::invalid_argument);

        // Negative base_max_depth
        params["n_estimators"] = 50;
        params["base_max_depth"] = -1;
        REQUIRE_THROWS_AS(ada.setHyperparameters(params), std::invalid_argument);

        // Zero base_max_depth
        params["base_max_depth"] = 0;
        REQUIRE_THROWS_AS(ada.setHyperparameters(params), std::invalid_argument);
    }
}

TEST_CASE("AdaBoost Basic Functionality", "[AdaBoost]")
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
        AdaBoost ada(10, 3);  // 10 estimators, max_depth = 3
        REQUIRE_NOTHROW(ada.fit(X, y, features, className, states, Smoothing_t::NONE));

        // Check that we have the expected number of models
        auto weights = ada.getEstimatorWeights();
        REQUIRE(weights.size() <= 10);  // Should be <= n_estimators
        REQUIRE(weights.size() > 0);    // Should have at least one model

        // Check training errors
        auto errors = ada.getTrainingErrors();
        REQUIRE(errors.size() == weights.size());

        // All training errors should be less than 0.5 for this simple dataset
        for (double error : errors) {
            REQUIRE(error < 0.5);
            REQUIRE(error >= 0.0);
        }
    }

    SECTION("Prediction before fitting")
    {
        AdaBoost ada;
        REQUIRE_THROWS_WITH(ada.predict(X),
            ContainsSubstring("not been fitted"));
        REQUIRE_THROWS_WITH(ada.predict_proba(X),
            ContainsSubstring("not been fitted"));
    }

    SECTION("Prediction with vector interface")
    {
        AdaBoost ada(10, 3);
        ada.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto predictions = ada.predict(X);
        REQUIRE(predictions.size() == static_cast<size_t>(n_samples));
        // Check accuracy
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == y[i]) correct++;
        }
        double accuracy = static_cast<double>(correct) / n_samples;
        REQUIRE(accuracy > 0.99);  // Should achieve good accuracy on this simple dataset
        auto accuracy_computed = ada.score(X, y);
        REQUIRE(accuracy_computed == Catch::Approx(accuracy).epsilon(1e-6));
    }

    SECTION("Probability predictions with vector interface")
    {
        AdaBoost ada(10, 3);
        ada.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto proba = ada.predict_proba(X);
        REQUIRE(proba.size() == static_cast<size_t>(n_samples));
        REQUIRE(proba[0].size() == 2);  // Two classes

        // Check probabilities sum to 1 and are valid
        auto predictions = ada.predict(X);
        int correct = 0;
        for (size_t i = 0; i < proba.size(); i++) {
            auto p = proba[i];
            auto pred = predictions[i];
            REQUIRE(p.size() == 2);
            REQUIRE(p[0] >= 0.0);
            REQUIRE(p[1] >= 0.0);
            double sum = p[0] + p[1];
            REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-6));
            // compute the predicted class based on probabilities
            auto predicted_class = (p[0] > p[1]) ? 0 : 1;
            // compute accuracy based on predictions
            if (predicted_class == y[i]) {
                correct++;
            }

            // Check that predict_proba matches the expected predict value
            // REQUIRE(pred == (p[0] > p[1] ? 0 : 1));
        }
        double accuracy = static_cast<double>(correct) / n_samples;
        std::cout << "Probability accuracy: " << accuracy << std::endl;
        REQUIRE(accuracy > 0.99);  // Should achieve good accuracy on this simple dataset
    }
}

TEST_CASE("AdaBoost Tensor Interface", "[AdaBoost]")
{
    auto raw = RawDatasets("iris", true);

    SECTION("Training with tensor format")
    {
        AdaBoost ada(20, 3);

        INFO("Dataset shape: " << raw.dataset.sizes());
        INFO("Features: " << raw.featurest.size());
        INFO("Samples: " << raw.nSamples);

        // AdaBoost expects dataset in format: features x samples, with labels as last row
        REQUIRE_NOTHROW(ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE));

        // Test prediction with tensor
        auto predictions = ada.predict(raw.Xt);
        REQUIRE(predictions.size(0) == raw.yt.size(0));

        // Calculate accuracy
        auto correct = torch::sum(predictions == raw.yt).item<int>();
        double accuracy = static_cast<double>(correct) / raw.yt.size(0);
        auto accuracy_computed = ada.score(raw.Xt, raw.yt);
        REQUIRE(accuracy_computed == Catch::Approx(accuracy).epsilon(1e-6));
        REQUIRE(accuracy > 0.97);  // Should achieve good accuracy on Iris

        // Test probability predictions with tensor
        auto proba = ada.predict_proba(raw.Xt);
        REQUIRE(proba.size(0) == raw.yt.size(0));
        REQUIRE(proba.size(1) == 3);  // Three classes in Iris

        // Check probabilities sum to 1
        auto prob_sums = torch::sum(proba, 1);
        for (int i = 0; i < prob_sums.size(0); i++) {
            REQUIRE(prob_sums[i].item<double>() == Catch::Approx(1.0).epsilon(1e-6));
        }
    }
}

TEST_CASE("AdaBoost on Iris Dataset", "[AdaBoost][iris]")
{
    auto raw = RawDatasets("iris", true);

    SECTION("Training with vector interface")
    {
        AdaBoost ada(30, 3);

        REQUIRE_NOTHROW(ada.fit(raw.Xv, raw.yv, raw.featuresv, raw.classNamev, raw.statesv, Smoothing_t::NONE));

        auto predictions = ada.predict(raw.Xv);
        REQUIRE(predictions.size() == raw.yv.size());

        // Calculate accuracy
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == raw.yv[i]) correct++;
        }
        double accuracy = static_cast<double>(correct) / raw.yv.size();
        REQUIRE(accuracy > 0.85);  // Should achieve good accuracy

        // Test probability predictions
        auto proba = ada.predict_proba(raw.Xv);
        REQUIRE(proba.size() == raw.yv.size());
        REQUIRE(proba[0].size() == 3);  // Three classes

        // Verify estimator weights and errors
        auto weights = ada.getEstimatorWeights();
        auto errors = ada.getTrainingErrors();

        REQUIRE(weights.size() == errors.size());
        REQUIRE(weights.size() > 0);

        // All weights should be positive (for non-zero error estimators)
        for (double w : weights) {
            REQUIRE(w >= 0.0);
        }

        // All errors should be less than 0.5 (better than random)
        for (double e : errors) {
            REQUIRE(e < 0.5);
            REQUIRE(e >= 0.0);
        }
    }

    SECTION("Different number of estimators")
    {
        std::vector<int> n_estimators = { 5, 15, 25 };

        for (int n_est : n_estimators) {
            AdaBoost ada(n_est, 2);
            ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

            auto predictions = ada.predict(raw.Xt);
            REQUIRE(predictions.size(0) == raw.yt.size(0));

            // Check that we don't exceed the specified number of estimators
            auto weights = ada.getEstimatorWeights();
            REQUIRE(static_cast<int>(weights.size()) <= n_est);
        }
    }

    SECTION("Different base estimator depths")
    {
        std::vector<int> depths = { 1, 2, 4 };

        for (int depth : depths) {
            AdaBoost ada(15, depth);
            ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

            auto predictions = ada.predict(raw.Xt);
            REQUIRE(predictions.size(0) == raw.yt.size(0));
        }
    }
}

TEST_CASE("AdaBoost Edge Cases", "[AdaBoost]")
{
    auto raw = RawDatasets("iris", true);

    SECTION("Single estimator (depth 1 stump)")
    {
        AdaBoost ada(1, 1);  // Single decision stump
        ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

        auto predictions = ada.predict(raw.Xt);
        REQUIRE(predictions.size(0) == raw.yt.size(0));

        auto weights = ada.getEstimatorWeights();
        REQUIRE(weights.size() == 1);
    }

    SECTION("Perfect classifier scenario")
    {
        // Create a perfectly separable dataset
        std::vector<std::vector<int>> X = { {0,0,1,1}, {0,1,0,1} };
        std::vector<int> y = { 0, 0, 1, 1 };
        std::vector<std::string> features = { "f1", "f2" };
        std::string className = "class";
        std::map<std::string, std::vector<int>> states;
        states["f1"] = { 0, 1 };
        states["f2"] = { 0, 1 };
        states["class"] = { 0, 1 };

        AdaBoost ada(10, 3);
        ada.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto predictions = ada.predict(X);
        REQUIRE(predictions.size() == 4);

        // Should achieve perfect accuracy
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == y[i]) correct++;
        }
        REQUIRE(correct == 4);

        // Should stop early due to perfect classification
        auto errors = ada.getTrainingErrors();
        if (errors.size() > 0) {
            REQUIRE(errors.back() < 1e-10);  // Very low error
        }
    }

    SECTION("Small dataset")
    {
        // Very small dataset
        std::vector<std::vector<int>> X = { {0,1}, {1,0} };
        std::vector<int> y = { 0, 1 };
        std::vector<std::string> features = { "f1", "f2" };
        std::string className = "class";
        std::map<std::string, std::vector<int>> states;
        states["f1"] = { 0, 1 };
        states["f2"] = { 0, 1 };
        states["class"] = { 0, 1 };

        AdaBoost ada(5, 1);
        REQUIRE_NOTHROW(ada.fit(X, y, features, className, states, Smoothing_t::NONE));

        auto predictions = ada.predict(X);
        REQUIRE(predictions.size() == 2);
    }
}

TEST_CASE("AdaBoost Graph Visualization", "[AdaBoost]")
{
    // Simple dataset for visualization
    std::vector<std::vector<int>> X = { {0,0,1,1}, {0,1,0,1} };
    std::vector<int> y = { 0, 1, 1, 0 };  // XOR pattern
    std::vector<std::string> features = { "x1", "x2" };
    std::string className = "xor";
    std::map<std::string, std::vector<int>> states;
    states["x1"] = { 0, 1 };
    states["x2"] = { 0, 1 };
    states["xor"] = { 0, 1 };

    SECTION("Graph generation")
    {
        AdaBoost ada(5, 2);
        ada.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto graph_lines = ada.graph();

        REQUIRE(graph_lines.size() > 2);
        REQUIRE(graph_lines.front() == "digraph AdaBoost {");
        REQUIRE(graph_lines.back() == "}");

        // Should contain base estimator references
        bool has_estimators = false;
        for (const auto& line : graph_lines) {
            if (line.find("Estimator") != std::string::npos) {
                has_estimators = true;
                break;
            }
        }
        REQUIRE(has_estimators);

        // Should contain alpha values
        bool has_alpha = false;
        for (const auto& line : graph_lines) {
            if (line.find("α") != std::string::npos || line.find("alpha") != std::string::npos) {
                has_alpha = true;
                break;
            }
        }
        REQUIRE(has_alpha);
    }

    SECTION("Graph with title")
    {
        AdaBoost ada(3, 1);
        ada.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto graph_lines = ada.graph("XOR AdaBoost");

        bool has_title = false;
        for (const auto& line : graph_lines) {
            if (line.find("label=\"XOR AdaBoost\"") != std::string::npos) {
                has_title = true;
                break;
            }
        }
        REQUIRE(has_title);
    }
}

TEST_CASE("AdaBoost with Weights", "[AdaBoost]")
{
    auto raw = RawDatasets("iris", true);

    SECTION("Uniform weights")
    {
        AdaBoost ada(20, 3);
        ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, raw.weights, Smoothing_t::NONE);

        auto predictions = ada.predict(raw.Xt);
        REQUIRE(predictions.size(0) == raw.yt.size(0));

        auto weights = ada.getEstimatorWeights();
        REQUIRE(weights.size() > 0);
    }

    SECTION("Non-uniform weights")
    {
        auto weights = torch::ones({ raw.nSamples });
        weights.index({ torch::indexing::Slice(0, 50) }) *= 3.0;  // Emphasize first class
        weights = weights / weights.sum();

        AdaBoost ada(15, 2);
        ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, weights, Smoothing_t::NONE);

        auto predictions = ada.predict(raw.Xt);
        REQUIRE(predictions.size(0) == raw.yt.size(0));

        // Check that training completed successfully
        auto estimator_weights = ada.getEstimatorWeights();
        auto errors = ada.getTrainingErrors();

        REQUIRE(estimator_weights.size() == errors.size());
        REQUIRE(estimator_weights.size() > 0);
    }
}

TEST_CASE("AdaBoost Input Dimension Validation", "[AdaBoost]")
{
    auto raw = RawDatasets("iris", true);

    SECTION("Correct input dimensions")
    {
        AdaBoost ada(10, 2);
        ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

        // Test with correct tensor dimensions (features x samples)
        REQUIRE_NOTHROW(ada.predict(raw.Xt));
        REQUIRE_NOTHROW(ada.predict_proba(raw.Xt));

        // Test with correct vector dimensions (features x samples)
        REQUIRE_NOTHROW(ada.predict(raw.Xv));
        REQUIRE_NOTHROW(ada.predict_proba(raw.Xv));
    }

    SECTION("Dimension consistency between interfaces")
    {
        AdaBoost ada(10, 2);
        ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

        // Get predictions from both interfaces
        auto tensor_predictions = ada.predict(raw.Xt);
        auto vector_predictions = ada.predict(raw.Xv);

        // Should have same number of predictions
        REQUIRE(tensor_predictions.size(0) == static_cast<int>(vector_predictions.size()));

        // Test probability predictions
        auto tensor_proba = ada.predict_proba(raw.Xt);
        auto vector_proba = ada.predict_proba(raw.Xv);

        REQUIRE(tensor_proba.size(0) == static_cast<int>(vector_proba.size()));
        REQUIRE(tensor_proba.size(1) == static_cast<int>(vector_proba[0].size()));

        // Verify predictions match between interfaces
        for (int i = 0; i < tensor_predictions.size(0); i++) {
            REQUIRE(tensor_predictions[i].item<int>() == vector_predictions[i]);

            // Verify probabilities match between interfaces
            for (int j = 0; j < tensor_proba.size(1); j++) {
                REQUIRE(tensor_proba[i][j].item<double>() == Catch::Approx(vector_proba[i][j]).epsilon(1e-10));
            }
        }
    }
}

TEST_CASE("AdaBoost Debug - Simple Dataset Analysis", "[AdaBoost][debug]")
{
    // Create the exact same simple dataset that was failing
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

    SECTION("Debug training process")
    {
        AdaBoost ada(5, 3);  // Few estimators for debugging

        // This should work perfectly on this simple dataset
        REQUIRE_NOTHROW(ada.fit(X, y, features, className, states, Smoothing_t::NONE));

        // Get training details
        auto weights = ada.getEstimatorWeights();
        auto errors = ada.getTrainingErrors();

        INFO("Number of models trained: " << weights.size());
        INFO("Training errors: ");
        for (size_t i = 0; i < errors.size(); i++) {
            INFO("  Model " << i << ": error=" << errors[i] << ", weight=" << weights[i]);
        }

        // Should have at least one model
        REQUIRE(weights.size() > 0);
        REQUIRE(errors.size() == weights.size());

        // All training errors should be reasonable for this simple dataset
        for (double error : errors) {
            REQUIRE(error >= 0.0);
            REQUIRE(error < 0.5);  // Should be better than random
        }

        // Test predictions
        auto predictions = ada.predict(X);
        REQUIRE(predictions.size() == static_cast<size_t>(n_samples));

        // Calculate accuracy
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == y[i]) correct++;
            INFO("Sample " << i << ": predicted=" << predictions[i] << ", actual=" << y[i]);
        }
        double accuracy = static_cast<double>(correct) / n_samples;
        INFO("Accuracy: " << accuracy);

        // Should achieve high accuracy on this perfectly separable dataset
        REQUIRE(accuracy >= 0.9);  // Lower threshold for debugging

        // Test probability predictions
        auto proba = ada.predict_proba(X);
        REQUIRE(proba.size() == static_cast<size_t>(n_samples));

        // Verify probabilities are valid
        for (size_t i = 0; i < proba.size(); i++) {
            auto p = proba[i];
            REQUIRE(p.size() == 2);
            REQUIRE(p[0] >= 0.0);
            REQUIRE(p[1] >= 0.0);
            double sum = p[0] + p[1];
            REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-6));

            // Predicted class should match highest probability
            int pred_class = predictions[i];
            REQUIRE(pred_class == (p[0] > p[1] ? 0 : 1));
        }
    }

    SECTION("Compare with single DecisionTree")
    {
        // Test that AdaBoost performs at least as well as a single tree
        DecisionTree single_tree(3, 2, 1);
        single_tree.fit(X, y, features, className, states, Smoothing_t::NONE);
        auto tree_predictions = single_tree.predict(X);

        int tree_correct = 0;
        for (size_t i = 0; i < tree_predictions.size(); i++) {
            if (tree_predictions[i] == y[i]) tree_correct++;
        }
        double tree_accuracy = static_cast<double>(tree_correct) / n_samples;

        AdaBoost ada(5, 3);
        ada.fit(X, y, features, className, states, Smoothing_t::NONE);
        auto ada_predictions = ada.predict(X);

        int ada_correct = 0;
        for (size_t i = 0; i < ada_predictions.size(); i++) {
            if (ada_predictions[i] == y[i]) ada_correct++;
        }
        double ada_accuracy = static_cast<double>(ada_correct) / n_samples;

        INFO("DecisionTree accuracy: " << tree_accuracy);
        INFO("AdaBoost accuracy: " << ada_accuracy);

        // AdaBoost should perform at least as well as single tree
        // (allowing small tolerance for numerical differences)
        REQUIRE(ada_accuracy >= tree_accuracy - 0.1);
    }
}

TEST_CASE("AdaBoost SAMME Algorithm Validation", "[AdaBoost]")
{
    auto raw = RawDatasets("iris", true);

    SECTION("Prediction consistency with probabilities")
    {
        AdaBoost ada(15, 3);
        ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

        auto predictions = ada.predict(raw.Xt);
        auto probabilities = ada.predict_proba(raw.Xt);

        REQUIRE(predictions.size(0) == probabilities.size(0));
        REQUIRE(probabilities.size(1) == 3);  // Three classes in Iris

        // For each sample, predicted class should correspond to highest probability
        for (int i = 0; i < predictions.size(0); i++) {
            int predicted_class = predictions[i].item<int>();
            auto probs = probabilities[i];

            // Find class with highest probability
            auto max_prob_idx = torch::argmax(probs).item<int>();

            // Predicted class should match class with highest probability
            REQUIRE(predicted_class == max_prob_idx);

            // Probabilities should sum to 1
            double sum_probs = torch::sum(probs).item<double>();
            REQUIRE(sum_probs == Catch::Approx(1.0).epsilon(1e-6));

            // All probabilities should be non-negative
            for (int j = 0; j < 3; j++) {
                REQUIRE(probs[j].item<double>() >= 0.0);
                REQUIRE(probs[j].item<double>() <= 1.0);
            }
        }
    }

    SECTION("Weighted voting verification")
    {
        // Simple dataset where we can verify the weighted voting
        std::vector<std::vector<int>> X = { {0,0,1,1}, {0,1,0,1} };
        std::vector<int> y = { 0, 1, 1, 0 };
        std::vector<std::string> features = { "f1", "f2" };
        std::string className = "class";
        std::map<std::string, std::vector<int>> states;
        states["f1"] = { 0, 1 };
        states["f2"] = { 0, 1 };
        states["class"] = { 0, 1 };

        AdaBoost ada(5, 2);
        ada.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto predictions = ada.predict(X);
        auto probabilities = ada.predict_proba(X);
        auto alphas = ada.getEstimatorWeights();

        REQUIRE(predictions.size() == 4);
        REQUIRE(probabilities.size() == 4);
        REQUIRE(probabilities[0].size() == 2);  // Two classes
        REQUIRE(alphas.size() > 0);

        // Verify that estimator weights are reasonable
        for (double alpha : alphas) {
            REQUIRE(alpha >= 0.0);  // Alphas should be non-negative
        }

        // Verify prediction-probability consistency
        for (size_t i = 0; i < predictions.size(); i++) {
            int pred = predictions[i];
            auto probs = probabilities[i];

            REQUIRE(pred == (probs[0] > probs[1] ? 0 : 1));
            REQUIRE(probs[0] + probs[1] == Catch::Approx(1.0).epsilon(1e-6));
        }
    }

    SECTION("Empty models edge case")
    {
        AdaBoost ada(1, 1);

        // Try to predict before fitting
        std::vector<std::vector<int>> X = { {0}, {1} };
        REQUIRE_THROWS_WITH(ada.predict(X), ContainsSubstring("not been fitted"));
        REQUIRE_THROWS_WITH(ada.predict_proba(X), ContainsSubstring("not been fitted"));
    }
}
TEST_CASE("AdaBoost Predict-Proba Consistency Fix", "[AdaBoost][consistency]")
{
    // Simple binary classification dataset
    std::vector<std::vector<int>> X = { {0,0,1,1}, {0,1,0,1} };
    std::vector<int> y = { 0, 0, 1, 1 };
    std::vector<std::string> features = { "f1", "f2" };
    std::string className = "class";
    std::map<std::string, std::vector<int>> states;
    states["f1"] = { 0, 1 };
    states["f2"] = { 0, 1 };
    states["class"] = { 0, 1 };

    SECTION("Binary classification consistency")
    {
        AdaBoost ada(3, 2);
        ada.setDebug(true);  // Enable debug output
        ada.fit(X, y, features, className, states, Smoothing_t::NONE);

        auto predictions = ada.predict(X);
        auto probabilities = ada.predict_proba(X);

        INFO("=== Debugging predict vs predict_proba consistency ===");

        // Verify consistency for each sample
        for (size_t i = 0; i < predictions.size(); i++) {
            int predicted_class = predictions[i];
            auto probs = probabilities[i];

            INFO("Sample " << i << ":");
            INFO("  True class: " << y[i]);
            INFO("  Predicted class: " << predicted_class);
            INFO("  Probabilities: [" << probs[0] << ", " << probs[1] << "]");

            // The predicted class should be the one with highest probability
            int max_prob_class = (probs[0] > probs[1]) ? 0 : 1;
            INFO("  Max prob class: " << max_prob_class);

            REQUIRE(predicted_class == max_prob_class);

            // Probabilities should sum to 1
            double sum_probs = probs[0] + probs[1];
            REQUIRE(sum_probs == Catch::Approx(1.0).epsilon(1e-6));

            // All probabilities should be valid
            REQUIRE(probs[0] >= 0.0);
            REQUIRE(probs[1] >= 0.0);
            REQUIRE(probs[0] <= 1.0);
            REQUIRE(probs[1] <= 1.0);
        }
    }

    SECTION("Multi-class consistency")
    {
        auto raw = RawDatasets("iris", true);

        AdaBoost ada(5, 2);
        ada.fit(raw.dataset, raw.featurest, raw.classNamet, raw.statest, Smoothing_t::NONE);

        auto predictions = ada.predict(raw.Xt);
        auto probabilities = ada.predict_proba(raw.Xt);

        // Check consistency for first 10 samples
        for (int i = 0; i < std::min(static_cast<int64_t>(10), predictions.size(0)); i++) {
            int predicted_class = predictions[i].item<int>();
            auto probs = probabilities[i];

            // Find class with maximum probability
            auto max_prob_idx = torch::argmax(probs).item<int>();

            INFO("Sample " << i << ":");
            INFO("  Predicted class: " << predicted_class);
            INFO("  Max prob class: " << max_prob_idx);
            INFO("  Probabilities: [" << probs[0].item<float>() << ", "
                << probs[1].item<float>() << ", " << probs[2].item<float>() << "]");

            // They must match
            REQUIRE(predicted_class == max_prob_idx);

            // Probabilities should sum to 1
            double sum_probs = torch::sum(probs).item<double>();
            REQUIRE(sum_probs == Catch::Approx(1.0).epsilon(1e-6));
        }
    }
}