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

// Conditional debug macro for performance-critical sections
#define DEBUG_LOG(condition, ...) \
    do { \
        if (__builtin_expect((condition), 0)) { \
            std::cout << __VA_ARGS__ << std::endl; \
        } \
    } while(0)

namespace bayesnet {

    AdaBoost::AdaBoost(int n_estimators, int max_depth)
        : Ensemble(true), n_estimators(n_estimators), base_max_depth(max_depth), n(0), n_classes(0)
    {
        validHyperparameters = { "n_estimators", "base_max_depth" };
    }

    // Versión optimizada de buildModel - Reemplazar en AdaBoost.cpp:

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
            if (weights.size(0) != n_samples) {
                throw std::runtime_error("weights must have the same length as number of samples");
            }
            sample_weights = weights.clone();
            normalizeWeights();
        }

        // Conditional debug information (only when debug is enabled)
        DEBUG_LOG(debug, "Starting AdaBoost training with " << n_estimators << " estimators\n"
            << "Number of classes: " << n_classes << "\n"
            << "Number of features: " << n << "\n"
            << "Number of samples: " << n_samples);

        // Pre-compute random guess error threshold
        const double random_guess_error = 1.0 - (1.0 / static_cast<double>(n_classes));

        // Main AdaBoost training loop (SAMME algorithm)
        for (int iter = 0; iter < n_estimators; ++iter) {
            // Train base estimator with current sample weights
            auto estimator = trainBaseEstimator(sample_weights);

            // Calculate weighted error
            double weighted_error = calculateWeightedError(estimator.get(), sample_weights);
            training_errors.push_back(weighted_error);

            // According to SAMME, we need error < random_guess_error
            if (weighted_error >= random_guess_error) {
                DEBUG_LOG(debug, "Error >= random guess (" << random_guess_error << "), stopping");
                // If only one estimator and it's worse than random, keep it with zero weight
                if (models.empty()) {
                    models.push_back(std::move(estimator));
                    alphas.push_back(0.0);
                }
                break;  // Stop boosting
            }

            // Check for perfect classification BEFORE calculating alpha
            if (weighted_error <= 1e-10) {
                DEBUG_LOG(debug, "Perfect classification achieved (error=" << weighted_error << ")");

                // For perfect classification, use a large but finite alpha
                double alpha = 10.0 + std::log(static_cast<double>(n_classes - 1));

                // Store the estimator and its weight
                models.push_back(std::move(estimator));
                alphas.push_back(alpha);

                DEBUG_LOG(debug, "Iteration " << iter << ":\n"
                    << "  Weighted error: " << weighted_error << "\n"
                    << "  Alpha (finite): " << alpha << "\n"
                    << "  Random guess error: " << random_guess_error);

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

            DEBUG_LOG(debug, "Iteration " << iter << ":\n"
                << "  Weighted error: " << weighted_error << "\n"
                << "  Alpha: " << alpha << "\n"
                << "  Random guess error: " << random_guess_error);
        }

        // Set the number of models actually trained
        n_models = models.size();
        DEBUG_LOG(debug, "AdaBoost training completed with " << n_models << " models");
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
        // Get features and labels from dataset (avoid repeated indexing)
        auto X = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), torch::indexing::Slice() });
        auto y_true = dataset.index({ -1, torch::indexing::Slice() });

        // Get predictions from the estimator
        auto y_pred = estimator->predict(X);

        // Vectorized error calculation using PyTorch operations
        auto incorrect = (y_pred != y_true).to(torch::kDouble);

        // Direct dot product for weighted error (more efficient than sum)
        double weighted_error = torch::dot(incorrect, weights).item<double>();

        // Clamp to valid range in one operation
        return std::clamp(weighted_error, 1e-15, 1.0 - 1e-15);
    }

    void AdaBoost::updateSampleWeights(Classifier* estimator, double alpha)
    {
        // Get predictions from the estimator (reuse from calculateWeightedError if possible)
        auto X = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), torch::indexing::Slice() });
        auto y_true = dataset.index({ -1, torch::indexing::Slice() });
        auto y_pred = estimator->predict(X);

        // Vectorized weight update using PyTorch operations
        auto incorrect = (y_pred != y_true).to(torch::kDouble);

        // Single vectorized operation instead of element-wise multiplication
        sample_weights *= torch::exp(alpha * incorrect);

        // Vectorized clamping for numerical stability
        sample_weights = torch::clamp(sample_weights, 1e-15, 1e15);
    }

    void AdaBoost::normalizeWeights()
    {
        // Single-pass normalization using PyTorch operations
        double sum_weights = torch::sum(sample_weights).item<double>();

        if (__builtin_expect(sum_weights <= 0, 0)) {
            // Reset to uniform if all weights are zero/negative (rare case)
            sample_weights = torch::ones_like(sample_weights) / sample_weights.size(0);
        } else {
            // Vectorized normalization
            sample_weights /= sum_weights;

            // Vectorized minimum weight enforcement
            sample_weights = torch::clamp_min(sample_weights, 1e-15);

            // Renormalize after clamping (if any weights were clamped)
            double new_sum = torch::sum(sample_weights).item<double>();
            if (new_sum != 1.0) {
                sample_weights /= new_sum;
            }
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

        if (debug) {
            std::cout << "=== predict_proba vector method debug ===" << std::endl;
            std::cout << "Input X dimensions: " << X.size() << " features x " << n_samples << " samples" << std::endl;
            std::cout << "Input data:" << std::endl;
            for (size_t i = 0; i < X.size(); i++) {
                std::cout << "  Feature " << i << ": [";
                for (size_t j = 0; j < X[i].size(); j++) {
                    std::cout << X[i][j];
                    if (j < X[i].size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }

        // Convert to tensor - X is features x samples, need to transpose for tensor format
        torch::Tensor X_tensor = platform::TensorUtils::to_matrix(X);

        if (debug) {
            std::cout << "Converted tensor shape: " << X_tensor.sizes() << std::endl;
            std::cout << "Tensor data: " << X_tensor << std::endl;
        }

        auto proba_tensor = predict_proba(X_tensor);  // Call tensor method

        if (debug) {
            std::cout << "Proba tensor shape: " << proba_tensor.sizes() << std::endl;
            std::cout << "Proba tensor data: " << proba_tensor << std::endl;
        }

        std::vector<std::vector<double>> result(n_samples, std::vector<double>(n_classes, 0.0));

        for (size_t i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_classes; j++) {
                result[i][j] = proba_tensor[i][j].item<double>();
            }

            if (debug) {
                std::cout << "Sample " << i << " converted: [";
                for (int j = 0; j < n_classes; j++) {
                    std::cout << result[i][j];
                    if (j < n_classes - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }

        if (debug) {
            std::cout << "=== End predict_proba vector method debug ===" << std::endl;
        }

        return result;
    }

    // También agregar debug al método tensor predict_proba:

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

        if (debug) {
            std::cout << "=== predict_proba tensor method debug ===" << std::endl;
            std::cout << "Input tensor shape: " << X.sizes() << std::endl;
            std::cout << "Number of samples: " << n_samples << std::endl;
            std::cout << "Number of classes: " << n_classes << std::endl;
        }

        torch::Tensor probabilities = torch::zeros({ n_samples, n_classes });

        for (int i = 0; i < n_samples; i++) {
            auto sample = X.index({ torch::indexing::Slice(), i });

            if (debug) {
                std::cout << "Processing sample " << i << ": " << sample << std::endl;
            }

            auto sample_probs = predictProbaSample(sample);

            if (debug) {
                std::cout << "Sample " << i << " probabilities from predictProbaSample: " << sample_probs << std::endl;
            }

            probabilities[i] = sample_probs;

            if (debug) {
                std::cout << "Assigned to probabilities[" << i << "]: " << probabilities[i] << std::endl;
            }
        }

        if (debug) {
            std::cout << "Final probabilities tensor: " << probabilities << std::endl;
            std::cout << "=== End predict_proba tensor method debug ===" << std::endl;
        }

        return probabilities;
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

        // Initialize class votes with zeros  
        std::vector<double> class_votes(n_classes, 0.0);

        if (debug) {
            std::cout << "=== predictSample Debug ===" << std::endl;
            std::cout << "Number of models: " << models.size() << std::endl;
        }

        // Accumulate votes from all estimators (same logic as predictProbaSample)
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
                }
            }
            catch (const std::exception& e) {
                if (debug) std::cout << "Error in model " << i << ": " << e.what() << std::endl;
                continue;
            }
        }

        // Find class with maximum votes
        int best_class = 0;
        double max_votes = class_votes[0];

        for (int j = 1; j < n_classes; j++) {
            if (class_votes[j] > max_votes) {
                max_votes = class_votes[j];
                best_class = j;
            }
        }

        if (debug) {
            std::cout << "Class votes: [";
            for (int j = 0; j < n_classes; j++) {
                std::cout << class_votes[j];
                if (j < n_classes - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "Best class: " << best_class << " with " << max_votes << " votes" << std::endl;
            std::cout << "=== End predictSample Debug ===" << std::endl;
        }

        return best_class;
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
        torch::Tensor class_probs = torch::zeros({ n_classes }, torch::kDouble);

        if (total_votes > 0) {
            // Simple division to get probabilities
            for (int j = 0; j < n_classes; j++) {
                class_probs[j] = static_cast<double>(class_votes[j] / total_votes);
            }
        } else {
            // If no valid votes, uniform distribution
            if (debug) std::cout << "No valid votes, using uniform distribution" << std::endl;
            class_probs.fill_(1.0f / n_classes);
        }

        if (debug) {
            std::cout << "Final probabilities: [";
            for (int j = 0; j < n_classes; j++) {
                std::cout << class_probs[j].item<double>();
                if (j < n_classes - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "=== End predictProbaSample Debug ===" << std::endl;
        }

        return class_probs;
    }

} // namespace bayesnet