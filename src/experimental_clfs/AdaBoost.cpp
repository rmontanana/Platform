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
#include "TensorUtils.hpp"

namespace bayesnet {

    AdaBoost::AdaBoost(int n_estimators, int max_depth)
        : Ensemble(true), n_estimators(n_estimators), base_max_depth(max_depth), n(0), n_classes(0)
    {
        validHyperparameters = { "n_estimators", "base_max_depth" };
    }

    void AdaBoost::buildModel(const torch::Tensor& weights)
    {
        // Initialize variables
        models.clear();
        alphas.clear();
        training_errors.clear();

        // Initialize n (number of features) and n_classes
        n = dataset.size(0) - 1;  // Exclude the label row
        n_classes = states[className].size();

        // Initialize sample weights uniformly
        int n_samples = dataset.size(1);
        sample_weights = torch::ones({ n_samples }) / n_samples;

        // If initial weights are provided, incorporate them
        if (weights.defined() && weights.numel() > 0) {
            sample_weights *= weights;
            normalizeWeights();
        }

        // Debug information
        if (debug) {
            std::cout << "Starting AdaBoost training with " << n_estimators << " estimators" << std::endl;
            std::cout << "Number of classes: " << n_classes << std::endl;
            std::cout << "Number of features: " << n << std::endl;
            std::cout << "Number of samples: " << n_samples << std::endl;
        }

        // Main AdaBoost training loop (SAMME algorithm) 
        // (Stagewise Additive Modeling using a Multi - class Exponential loss)
        for (int iter = 0; iter < n_estimators; ++iter) {
            // Train base estimator with current sample weights
            auto estimator = trainBaseEstimator(sample_weights);

            // Calculate weighted error
            double weighted_error = calculateWeightedError(estimator.get(), sample_weights);
            training_errors.push_back(weighted_error);

            // Check if error is too high (worse than random guessing)
            double random_guess_error = 1.0 - (1.0 / n_classes);

            // According to SAMME, we need error < random_guess_error
            if (weighted_error >= random_guess_error) {
                if (debug) std::cout << "  Error >= random guess (" << random_guess_error << "), stopping" << std::endl;
                // If only one estimator and it's worse than random, keep it with zero weight
                if (models.empty()) {
                    models.push_back(std::move(estimator));
                    alphas.push_back(0.0);
                }
                break;  // Stop boosting
            }

            // Check for perfect classification BEFORE calculating alpha
            if (weighted_error <= 1e-10) {
                if (debug) std::cout << "  Perfect classification achieved (error=" << weighted_error << ")" << std::endl;

                // For perfect classification, use a large but finite alpha
                double alpha = 10.0 + std::log(static_cast<double>(n_classes - 1));

                // Store the estimator and its weight
                models.push_back(std::move(estimator));
                alphas.push_back(alpha);

                if (debug) {
                    std::cout << "Iteration " << iter << ":" << std::endl;
                    std::cout << "  Weighted error: " << weighted_error << std::endl;
                    std::cout << "  Alpha (finite): " << alpha << std::endl;
                    std::cout << "  Random guess error: " << random_guess_error << std::endl;
                }

                break;  // Stop training as we have a perfect classifier
            }

            // Calculate alpha (estimator weight) using SAMME formula
            // alpha = log((1 - err) / err) + log(K - 1)
            // Clamp weighted_error to avoid division by zero and infinite alpha
            double clamped_error = std::max(1e-15, std::min(1.0 - 1e-15, weighted_error));
            double alpha = std::log((1.0 - clamped_error) / clamped_error) +
                std::log(static_cast<double>(n_classes - 1));

            // Clamp alpha to reasonable bounds to avoid numerical issues
            alpha = std::max(-10.0, std::min(10.0, alpha));

            // Store the estimator and its weight
            models.push_back(std::move(estimator));
            alphas.push_back(alpha);

            // Update sample weights (only if this is not the last iteration)
            if (iter < n_estimators - 1) {
                updateSampleWeights(models.back().get(), alpha);
                normalizeWeights();
            }

            if (debug) {
                std::cout << "Iteration " << iter << ":" << std::endl;
                std::cout << "  Weighted error: " << weighted_error << std::endl;
                std::cout << "  Alpha: " << alpha << std::endl;
                std::cout << "  Random guess error: " << random_guess_error << std::endl;
                std::cout << "  Random guess error: " << random_guess_error << std::endl;
            }
        }

        // Set the number of models actually trained
        n_models = models.size();
        if (debug) std::cout << "AdaBoost training completed with " << n_models << " models" << std::endl;
    }

    void AdaBoost::trainModel(const torch::Tensor& weights, const Smoothing_t smoothing)
    {
        // Call buildModel which does the actual training
        buildModel(weights);
        fitted = true;
    }

    std::unique_ptr<Classifier> AdaBoost::trainBaseEstimator(const torch::Tensor& weights)
    {
        // Create a decision tree with specified max depth
        auto tree = std::make_unique<DecisionTree>(base_max_depth);

        // Ensure weights are properly normalized
        auto normalized_weights = weights / weights.sum();

        // Fit the tree with the current sample weights
        tree->fit(dataset, features, className, states, normalized_weights, Smoothing_t::NONE);

        return tree;
    }

    double AdaBoost::calculateWeightedError(Classifier* estimator, const torch::Tensor& weights)
    {
        // Get features and labels from dataset
        auto X = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), torch::indexing::Slice() });
        auto y_true = dataset.index({ -1, torch::indexing::Slice() });

        // Get predictions from the estimator
        auto y_pred = estimator->predict(X);

        // Calculate weighted error
        auto incorrect = (y_pred != y_true).to(torch::kFloat);

        // Ensure weights are normalized
        auto normalized_weights = weights / weights.sum();

        // Calculate weighted error
        double weighted_error = torch::sum(incorrect * normalized_weights).item<double>();

        return weighted_error;
    }

    void AdaBoost::updateSampleWeights(Classifier* estimator, double alpha)
    {
        // Get predictions from the estimator
        auto X = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), torch::indexing::Slice() });
        auto y_true = dataset.index({ -1, torch::indexing::Slice() });
        auto y_pred = estimator->predict(X);

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

    void AdaBoost::checkValues() const
    {
        if (n_estimators <= 0) {
            throw std::invalid_argument("n_estimators must be positive");
        }
        if (base_max_depth <= 0) {
            throw std::invalid_argument("base_max_depth must be positive");
        }
    }

    void AdaBoost::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        // Set hyperparameters from JSON
        auto it = hyperparameters.find("n_estimators");
        if (it != hyperparameters.end()) {
            n_estimators = it->get<int>();
            hyperparameters.erase("n_estimators");
        }

        it = hyperparameters.find("base_max_depth");
        if (it != hyperparameters.end()) {
            base_max_depth = it->get<int>();
            hyperparameters.erase("base_max_depth");
        }
        checkValues();
        Ensemble::setHyperparameters(hyperparameters);
    }

    torch::Tensor AdaBoost::predict(torch::Tensor& X)
    {
        if (!fitted) {
            throw std::runtime_error(CLASSIFIER_NOT_FITTED);
        }

        if (models.empty()) {
            throw std::runtime_error("No models have been trained");
        }

        // X should be (n_features, n_samples)
        if (X.size(0) != n) {
            throw std::runtime_error("Input has wrong number of features. Expected " +
                std::to_string(n) + " but got " + std::to_string(X.size(0)));
        }

        int n_samples = X.size(1);
        torch::Tensor predictions = torch::zeros({ n_samples }, torch::kInt32);

        for (int i = 0; i < n_samples; i++) {
            auto sample = X.index({ torch::indexing::Slice(), i });
            predictions[i] = predictSample(sample);
        }

        return predictions;
    }

    torch::Tensor AdaBoost::predict_proba(torch::Tensor& X)
    {
        if (!fitted) {
            throw std::runtime_error(CLASSIFIER_NOT_FITTED);
        }

        if (models.empty()) {
            throw std::runtime_error("No models have been trained");
        }

        // X should be (n_features, n_samples)
        if (X.size(0) != n) {
            throw std::runtime_error("Input has wrong number of features. Expected " +
                std::to_string(n) + " but got " + std::to_string(X.size(0)));
        }

        int n_samples = X.size(1);
        torch::Tensor probabilities = torch::zeros({ n_samples, n_classes });

        for (int i = 0; i < n_samples; i++) {
            auto sample = X.index({ torch::indexing::Slice(), i });
            probabilities[i] = predictProbaSample(sample);
        }

        return probabilities;
    }

    std::vector<int> AdaBoost::predict(std::vector<std::vector<int>>& X)
    {
        // Convert to tensor - X is samples x features, need to transpose
        torch::Tensor X_tensor = platform::TensorUtils::to_matrix(X);
        auto predictions = predict(X_tensor);
        std::vector<int> result = platform::TensorUtils::to_vector<int>(predictions);
        return result;
    }

    std::vector<std::vector<double>> AdaBoost::predict_proba(std::vector<std::vector<int>>& X)
    {
        auto n_samples = X[0].size();
        // Convert to tensor - X is samples x features, need to transpose
        torch::Tensor X_tensor = platform::TensorUtils::to_matrix(X);
        auto proba_tensor = predict_proba(X_tensor);

        std::vector<std::vector<double>> result(n_samples, std::vector<double>(n_classes, 0.0));

        for (size_t i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_classes; j++) {
                result[i][j] = proba_tensor[i][j].item<double>();
            }
        }

        return result;
    }

    int AdaBoost::predictSample(const torch::Tensor& x) const
    {
        if (!fitted) {
            throw std::runtime_error(CLASSIFIER_NOT_FITTED);
        }

        if (models.empty()) {
            throw std::runtime_error("No models have been trained");
        }

        // x should be a 1D tensor with n features
        if (x.size(0) != n) {
            throw std::runtime_error("Input sample has wrong number of features. Expected " +
                std::to_string(n) + " but got " + std::to_string(x.size(0)));
        }

        // Initialize class votes
        std::vector<double> class_votes(n_classes, 0.0);

        // Accumulate weighted votes from all estimators
        for (size_t i = 0; i < models.size(); i++) {
            if (alphas[i] <= 0) continue;  // Skip estimators with zero or negative weight
            try {
                // Get prediction from this estimator
                int predicted_class = static_cast<DecisionTree*>(models[i].get())->predictSample(x);

                // Add weighted vote for this class
                if (predicted_class >= 0 && predicted_class < n_classes) {
                    class_votes[predicted_class] += alphas[i];
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error in estimator " << i << ": " << e.what() << std::endl;
                continue;
            }
        }

        // Return class with highest weighted vote
        return std::distance(class_votes.begin(),
            std::max_element(class_votes.begin(), class_votes.end()));
    }

    torch::Tensor AdaBoost::predictProbaSample(const torch::Tensor& x) const
    {
        if (!fitted) {
            throw std::runtime_error(CLASSIFIER_NOT_FITTED);
        }

        if (models.empty()) {
            throw std::runtime_error("No models have been trained");
        }

        // x should be a 1D tensor with n features
        if (x.size(0) != n) {
            throw std::runtime_error("Input sample has wrong number of features. Expected " +
                std::to_string(n) + " but got " + std::to_string(x.size(0)));
        }

        // Initialize class votes with zeros
        std::vector<double> class_votes(n_classes, 0.0);
        double total_votes = 0.0;

        if (debug) {
            std::cout << "=== predictProbaSample Debug ===" << std::endl;
            std::cout << "Number of models: " << models.size() << std::endl;
            std::cout << "Number of classes: " << n_classes << std::endl;
        }

        // Accumulate votes from all estimators
        for (size_t i = 0; i < models.size(); i++) {
            double alpha = alphas[i];

            // Skip invalid estimators
            if (alpha <= 0 || !std::isfinite(alpha)) {
                if (debug) std::cout << "Skipping model " << i << " (alpha=" << alpha << ")" << std::endl;
                continue;
            }

            try {
                // Get class prediction from this estimator
                int predicted_class = static_cast<DecisionTree*>(models[i].get())->predictSample(x);

                if (debug) {
                    std::cout << "Model " << i << ": predicts class " << predicted_class
                        << " with alpha " << alpha << std::endl;
                }

                // Add weighted vote for this class
                if (predicted_class >= 0 && predicted_class < n_classes) {
                    class_votes[predicted_class] += alpha;
                    total_votes += alpha;
                } else {
                    if (debug) std::cout << "Invalid class prediction: " << predicted_class << std::endl;
                }
            }
            catch (const std::exception& e) {
                if (debug) std::cout << "Error in model " << i << ": " << e.what() << std::endl;
                continue;
            }
        }

        if (debug) {
            std::cout << "Total votes: " << total_votes << std::endl;
            std::cout << "Class votes: [";
            for (int j = 0; j < n_classes; j++) {
                std::cout << class_votes[j];
                if (j < n_classes - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Convert votes to probabilities
        torch::Tensor class_probs = torch::zeros({ n_classes }, torch::kFloat);

        if (total_votes > 0) {
            // Simple division to get probabilities
            for (int j = 0; j < n_classes; j++) {
                class_probs[j] = static_cast<float>(class_votes[j] / total_votes);
            }
        } else {
            // If no valid votes, uniform distribution
            if (debug) std::cout << "No valid votes, using uniform distribution" << std::endl;
            class_probs.fill_(1.0f / n_classes);
        }

        if (debug) {
            std::cout << "Final probabilities: [";
            for (int j = 0; j < n_classes; j++) {
                std::cout << class_probs[j].item<float>();
                if (j < n_classes - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "=== End predictProbaSample Debug ===" << std::endl;
        }

        return class_probs;
    }

} // namespace bayesnet