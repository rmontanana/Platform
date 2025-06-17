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
        void setNEstimators(int n_estimators) { this->n_estimators = n_estimators; }
        int getNEstimators() const { return n_estimators; }
        void setBaseMaxDepth(int depth) { this->base_max_depth = depth; }
        int getBaseMaxDepth() const { return base_max_depth; }

        // Get the weight of each base estimator
        std::vector<double> getEstimatorWeights() const { return alphas; }

        // Get training errors for each iteration
        std::vector<double> getTrainingErrors() const { return training_errors; }

        // Override setHyperparameters from BaseClassifier
        void setHyperparameters(const nlohmann::json& hyperparameters) override;

    protected:
        void buildModel(const torch::Tensor& weights) override;
        void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) override;

    private:
        int n_estimators;
        int base_max_depth;  // Max depth for base decision trees
        std::vector<double> alphas;  // Weight of each base estimator
        std::vector<double> training_errors;  // Training error at each iteration
        torch::Tensor sample_weights;  // Current sample weights

        // Train a single base estimator
        std::unique_ptr<Classifier> trainBaseEstimator(const torch::Tensor& weights);

        // Calculate weighted error
        double calculateWeightedError(Classifier* estimator, const torch::Tensor& weights);

        // Update sample weights based on predictions
        void updateSampleWeights(Classifier* estimator, double alpha);

        // Normalize weights to sum to 1
        void normalizeWeights();
    };
}

#endif // ADABOOST_H