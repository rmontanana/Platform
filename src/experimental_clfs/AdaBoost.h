// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>
#include <memory>
#include "bayesnet/ensembles/Ensemble.h"

namespace bayesnet {
    class AdaBoost : public Ensemble {
    public:
        explicit AdaBoost(int n_estimators = 50, int max_depth = 1);
        virtual ~AdaBoost() = default;

        // Override base class methods
        std::vector<std::string> graph(const std::string& title = "") const override;

        // AdaBoost specific methods
        void setNEstimators(int n_estimators) { this->n_estimators = n_estimators; checkValues(); }
        int getNEstimators() const { return n_estimators; }
        void setBaseMaxDepth(int depth) { this->base_max_depth = depth; checkValues(); }
        int getBaseMaxDepth() const { return base_max_depth; }

        // Get the weight of each base estimator
        std::vector<double> getEstimatorWeights() const { return alphas; }

        // Get training errors for each iteration
        std::vector<double> getTrainingErrors() const { return training_errors; }

        // Override setHyperparameters from BaseClassifier
        void setHyperparameters(const nlohmann::json& hyperparameters) override;

        torch::Tensor predict(torch::Tensor& X) override;
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>>& X);
        void setDebug(bool debug) { this->debug = debug; }

    protected:
        void buildModel(const torch::Tensor& weights) override;
        void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) override;

    private:
        int n_estimators;
        int base_max_depth;  // Max depth for base decision trees
        std::vector<double> alphas;  // Weight of each base estimator
        std::vector<double> training_errors;  // Training error at each iteration
        torch::Tensor sample_weights;  // Current sample weights
        int n_classes;  // Number of classes in the target variable
        int n;  // Number of features

        // Train a single base estimator
        std::unique_ptr<Classifier> trainBaseEstimator(const torch::Tensor& weights);

        // Calculate weighted error
        double calculateWeightedError(Classifier* estimator, const torch::Tensor& weights);

        // Update sample weights based on predictions
        void updateSampleWeights(Classifier* estimator, double alpha);

        // Normalize weights to sum to 1
        void normalizeWeights();

        // Check if hyperparameters values are valid
        void checkValues() const;

        // Make predictions for a single sample
        int predictSample(const torch::Tensor& x) const;

        // Make probabilistic predictions for a single sample
        torch::Tensor predictProbaSample(const torch::Tensor& x) const;
        bool debug = false;  // Enable debug mode for debug output
    };
}

#endif // ADABOOST_H