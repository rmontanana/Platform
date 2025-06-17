// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "AdaBoost.h"
#include "DecisionTree.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace bayesnet {

    AdaBoost::AdaBoost(int n_estimators, int max_depth)
        : Ensemble(true), n_estimators(n_estimators), base_max_depth(max_depth)
    {
        validHyperparameters = { "n_estimators", "base_max_depth" };
    }

    void AdaBoost::buildModel(const torch::Tensor& weights)
    {
        // Initialize variables
        models.clear();
        alphas.clear();
        training_errors.clear();

        // Initialize sample weights uniformly
        int n_samples = dataset.size(1);
        sample_weights = torch::ones({ n_samples }) / n_samples;

        // If initial weights are provided, incorporate them
        if (weights.defined() && weights.numel() > 0) {
            sample_weights *= weights;
            normalizeWeights();
        }

        // Main AdaBoost training loop (SAMME algorithm)
        for (int iter = 0; iter < n_estimators; ++iter) {
            // Train base estimator with current sample weights
            auto estimator = trainBaseEstimator(sample_weights);

            // Calculate weighted error
            double weighted_error = calculateWeightedError(estimator.get(), sample_weights);
            training_errors.push_back(weighted_error);

            // Check if error is too high (worse than random guessing)
            double random_guess_error = 1.0 - (1.0 / getClassNumStates());
            if (weighted_error >= random_guess_error) {
                // If only one estimator and it's worse than random, keep it with zero weight
                if (models.empty()) {
                    models.push_back(std::move(estimator));
                    alphas.push_back(0.0);
                }
                break;  // Stop boosting
            }

            // Calculate alpha (estimator weight) using SAMME formula
            // alpha = log((1 - err) / err) + log(K - 1)
            double alpha = std::log((1.0 - weighted_error) / weighted_error) +
                std::log(static_cast<double>(getClassNumStates() - 1));

            // Store the estimator and its weight
            models.push_back(std::move(estimator));
            alphas.push_back(alpha);

            // Update sample weights
            updateSampleWeights(models.back().get(), alpha);

            // Normalize weights
            normalizeWeights();

            // Check for perfect classification
            if (weighted_error < 1e-10) {
                break;
            }
        }

        // Set the number of models actually trained
        n_models = models.size();
    }

    void AdaBoost::trainModel(const torch::Tensor& weights, const Smoothing_t smoothing)
    {
        // AdaBoost handles its own weight management, so we just build the model
        buildModel(weights);
    }

    std::unique_ptr<Classifier> AdaBoost::trainBaseEstimator(const torch::Tensor& weights)
    {
        // Create a decision tree with specified max depth
        // For AdaBoost, we typically use shallow trees (stumps with max_depth=1)
        auto tree = std::make_unique<DecisionTree>(base_max_depth);

        // Fit the tree with the current sample weights
        tree->fit(dataset, features, className, states, weights, Smoothing_t::NONE);

        return tree;
    }

    double AdaBoost::calculateWeightedError(Classifier* estimator, const torch::Tensor& weights)
    {
        // Get predictions from the estimator
        auto X = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), torch::indexing::Slice() });
        auto y_true = dataset.index({ -1, torch::indexing::Slice() });
        auto y_pred = estimator->predict(X.t());

        // Calculate weighted error
        auto incorrect = (y_pred != y_true).to(torch::kFloat);
        double weighted_error = torch::sum(incorrect * weights).item<double>();

        return weighted_error;
    }

    void AdaBoost::updateSampleWeights(Classifier* estimator, double alpha)
    {
        // Get predictions from the estimator
        auto X = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), torch::indexing::Slice() });
        auto y_true = dataset.index({ -1, torch::indexing::Slice() });
        auto y_pred = estimator->predict(X.t());

        // Update weights according to SAMME algorithm
        // w_i = w_i * exp(alpha * I(y_i != y_pred_i))
        auto incorrect = (y_pred != y_true).to(torch::kFloat);
        sample_weights *= torch::exp(alpha * incorrect);
    }

    void AdaBoost::normalizeWeights()
    {
        // Normalize weights to sum to 1
        double sum_weights = torch::sum(sample_weights).item<double>();
        if (sum_weights > 0) {
            sample_weights /= sum_weights;
        }
    }

    std::vector<std::string> AdaBoost::graph(const std::string& title) const
    {
        // Create a graph representation of the AdaBoost ensemble
        std::vector<std::string> graph_lines;

        // Header
        graph_lines.push_back("digraph AdaBoost {");
        graph_lines.push_back("    rankdir=TB;");
        graph_lines.push_back("    node [shape=box];");

        if (!title.empty()) {
            graph_lines.push_back("    label=\"" + title + "\";");
            graph_lines.push_back("    labelloc=t;");
        }

        // Add input node
        graph_lines.push_back("    Input [shape=ellipse, label=\"Input Features\"];");

        // Add base estimators
        for (size_t i = 0; i < models.size(); ++i) {
            std::stringstream ss;
            ss << "    Estimator" << i << " [label=\"Base Estimator " << i + 1
                << "\\nα = " << std::fixed << std::setprecision(3) << alphas[i] << "\"];";
            graph_lines.push_back(ss.str());

            // Connect input to estimator
            ss.str("");
            ss << "    Input -> Estimator" << i << ";";
            graph_lines.push_back(ss.str());
        }

        // Add combination node
        graph_lines.push_back("    Combination [shape=diamond, label=\"Weighted Vote\"];");

        // Connect estimators to combination
        for (size_t i = 0; i < models.size(); ++i) {
            std::stringstream ss;
            ss << "    Estimator" << i << " -> Combination;";
            graph_lines.push_back(ss.str());
        }

        // Add output node
        graph_lines.push_back("    Output [shape=ellipse, label=\"Final Prediction\"];");
        graph_lines.push_back("    Combination -> Output;");

        // Close graph
        graph_lines.push_back("}");

        return graph_lines;
    }

    void AdaBoost::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        // Set hyperparameters from JSON
        auto it = hyperparameters.find("n_estimators");
        if (it != hyperparameters.end()) {
            n_estimators = it->get<int>();
            if (n_estimators <= 0) {
                throw std::invalid_argument("n_estimators must be positive");
            }
            hyperparameters.erase("n_estimators");  // Remove 'n_estimators' if present 
        }

        it = hyperparameters.find("base_max_depth");
        if (it != hyperparameters.end()) {
            base_max_depth = it->get<int>();
            if (base_max_depth <= 0) {
                throw std::invalid_argument("base_max_depth must be positive");
            }
            hyperparameters.erase("base_max_depth");  // Remove 'base_max_depth' if present 
        }
        Ensemble::setHyperparameters(hyperparameters);
    }

} // namespace bayesnet