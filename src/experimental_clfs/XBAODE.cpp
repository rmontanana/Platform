// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
#include <random> 
#include <set>
#include <functional>
#include <limits.h>
#include <tuple>
#include "XBAODE.h"
#include "XSpode.hpp"
#include "TensorUtils.hpp"
#include <loguru.hpp>

namespace platform {
    XBAODE::XBAODE()
    {
        validHyperparameters = { "alpha_block", "order", "convergence", "convergence_best", "bisection", "threshold", "maxTolerance",
            "predict_voting", "select_features" };
    }
    void XBAODE::add_model(std::unique_ptr<XSpode> model)
    {
        models.push_back(std::move(model));
        n_models++;
    }
    void XBAODE::remove_last_model()
    {
        models.pop_back();
        n_models--;
    }
    void XBAODE::trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing)
    {
        fitted = true;
        X_train_ = TensorUtils::to_matrix(X_train);
        y_train_ = TensorUtils::to_vector<int>(y_train);
        X_test_ = TensorUtils::to_matrix(X_test);
        y_test_ = TensorUtils::to_vector<int>(y_test);
        maxTolerance = 3;
        //
        // Logging setup
        //
        // loguru::set_thread_name("XBAODE");
        // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
        // loguru::add_file("XBAODE.log", loguru::Truncate, loguru::Verbosity_MAX);

        // Algorithm based on the adaboost algorithm for classification
        // as explained in Ensemble methods (Zhi-Hua Zhou, 2012)
        double alpha_t = 0;
        weights_ = torch::full({ m }, 1.0 / static_cast<double>(m), torch::kFloat64); // m initialized in Classifier.cc
        significanceModels.resize(n, 0.0); // n initialized in Classifier.cc
        bool finished = false;
        std::vector<int> featuresUsed;
        n_models = 0;
        std::unique_ptr<XSpode> model;
        if (selectFeatures) {
            featuresUsed = featureSelection(weights_);
            for (const auto& parent : featuresUsed) {
                model = std::unique_ptr<XSpode>(new XSpode(parent));
                model->fit(X_train_, y_train_, weights_, smoothing);
                std::cout << model->getNFeatures() << std::endl;
                add_model(std::move(model));
            }
            notes.push_back("Used features in initialization: " + std::to_string(featuresUsed.size()) + " of " + std::to_string(features.size()) + " with " + select_features_algorithm);
            auto ypred = ExpEnsemble::predict(X_train);
            std::tie(weights_, alpha_t, finished) = update_weights(y_train, ypred, weights_);
            // Update significance of the models
            for (const auto& parent : featuresUsed) {
                significanceModels_[parent] = alpha_t;
            }
            n_models = featuresUsed.size();
            // VLOG_SCOPE_F(1, "SelectFeatures. alpha_t: %f n_models: %d", alpha_t, n_models);
            if (finished) {
                return;
            }
        }
        int numItemsPack = 0; // The counter of the models inserted in the current pack
        // Variables to control the accuracy finish condition
        double priorAccuracy = 0.0;
        double improvement = 1.0;
        double convergence_threshold = 1e-4;
        int tolerance = 0; // number of times the accuracy is lower than the convergence_threshold
        // Step 0: Set the finish condition
        // epsilon sub t > 0.5 => inverse the weights policy
        // validation error is not decreasing
        // run out of features
        bool ascending = order_algorithm == bayesnet::Orders.ASC;
        std::mt19937 g{ 173 };
        while (!finished) {
            // Step 1: Build ranking with mutual information
            auto featureSelection = metrics.SelectKBestWeighted(weights_, ascending, n); // Get all the features sorted
            if (order_algorithm == bayesnet::Orders.RAND) {
                std::shuffle(featureSelection.begin(), featureSelection.end(), g);
            }
            // Remove used features
            featureSelection.erase(remove_if(featureSelection.begin(), featureSelection.end(), [&](auto x)
                { return std::find(featuresUsed.begin(), featuresUsed.end(), x) != featuresUsed.end();}),
                featureSelection.end()
            );
            int k = bisection ? pow(2, tolerance) : 1;
            int counter = 0; // The model counter of the current pack
            // VLOG_SCOPE_F(1, "counter=%d k=%d featureSelection.size: %zu", counter, k, featureSelection.size());
            while (counter++ < k && featureSelection.size() > 0) {
                auto feature = featureSelection[0];
                featureSelection.erase(featureSelection.begin());
                model = std::unique_ptr<XSpode>(new XSpode(feature));
                model->fit(X_train_, y_train_, weights_, smoothing);
                std::vector<int> ypred;
                if (alpha_block) {
                    //
                    // Compute the prediction with the current ensemble + model
                    //
                    // Add the model to the ensemble
                    significanceModels[feature] = 1.0;
                    add_model(std::move(model));
                    // Compute the prediction
                    ypred = ExpEnsemble::predict(X_train_);
                    // Remove the model from the ensemble
                    significanceModels[feature] = 0.0;
                    model = std::move(models_.back());
                    remove_last_model();
                } else {
                    ypred = model->predict(X_train_);
                }
                // Step 3.1: Compute the classifier amout of say
                auto ypred_t = torch::tensor(ypred);
                std::tie(weights_, alpha_t, finished) = update_weights(y_train, ypred_t, weights_);
                // Step 3.4: Store classifier and its accuracy to weigh its future vote
                numItemsPack++;
                featuresUsed.push_back(feature);
                add_model(std::move(model));
                significanceModels[feature] = alpha_t;
                // VLOG_SCOPE_F(2, "finished: %d numItemsPack: %d n_models: %d featuresUsed: %zu", finished, numItemsPack, n_models, featuresUsed.size());
            } // End of the pack
            if (convergence && !finished) {
                auto y_val_predict = ExpEnsemble::predict(X_test);
                double accuracy = (y_val_predict == y_test).sum().item<double>() / (double)y_test.size(0);
                if (priorAccuracy == 0) {
                    priorAccuracy = accuracy;
                } else {
                    improvement = accuracy - priorAccuracy;
                }
                if (improvement < convergence_threshold) {
                    // VLOG_SCOPE_F(3, "  (improvement<threshold) tolerance: %d numItemsPack: %d improvement: %f prior: %f current: %f", tolerance, numItemsPack, improvement, priorAccuracy, accuracy);
                    tolerance++;
                } else {
                    // VLOG_SCOPE_F(3, "* (improvement>=threshold) Reset. tolerance: %d numItemsPack: %d improvement: %f prior: %f current: %f", tolerance, numItemsPack, improvement, priorAccuracy, accuracy);
                    tolerance = 0; // Reset the counter if the model performs better
                    numItemsPack = 0;
                }
                if (convergence_best) {
                    // Keep the best accuracy until now as the prior accuracy
                    priorAccuracy = std::max(accuracy, priorAccuracy);
                } else {
                    // Keep the last accuray obtained as the prior accuracy
                    priorAccuracy = accuracy;
                }
            }
            // VLOG_SCOPE_F(1, "tolerance: %d featuresUsed.size: %zu features.size: %zu", tolerance, featuresUsed.size(), features.size());
            finished = finished || tolerance > maxTolerance || featuresUsed.size() == features.size();
        }
        if (tolerance > maxTolerance) {
            if (numItemsPack < n_models) {
                notes.push_back("Convergence threshold reached & " + std::to_string(numItemsPack) + " models eliminated");
                // VLOG_SCOPE_F(4, "Convergence threshold reached & %d models eliminated of %d", numItemsPack, n_models);
                for (int i = featuresUsed.size() - 1; i >= featuresUsed.size() - numItemsPack; --i) {
                    remove_last_model();
                    significanceModels[featuresUsed[i]] = 0.0;
                }
                // VLOG_SCOPE_F(4, "*Convergence threshold %d models left & %d features used.", n_models, featuresUsed.size());
            } else {
                notes.push_back("Convergence threshold reached & 0 models eliminated");
                // VLOG_SCOPE_F(4, "Convergence threshold reached & 0 models eliminated n_models=%d numItemsPack=%d", n_models, numItemsPack);
            }
        }
        if (featuresUsed.size() != features.size()) {
            notes.push_back("Used features in train: " + std::to_string(featuresUsed.size()) + " of " + std::to_string(features.size()));
            status = bayesnet::WARNING;
        }
        notes.push_back("Number of models: " + std::to_string(n_models));
        return;
    }
}