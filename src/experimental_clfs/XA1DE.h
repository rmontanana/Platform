// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XA1DE_H
#define XA1DE_H
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include "bayesnet/BaseClassifier.h"
#include "common/Timer.hpp"
#include "CountingSemaphore.hpp"
#include "Xaode.hpp"

namespace platform {

    class XA1DE : public bayesnet::BaseClassifier {
    public:
        XA1DE();
        virtual ~XA1DE() = default;
        const std::string CLASSIFIER_NOT_FITTED = "Classifier has not been fitted";

        XA1DE& fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) override;
        XA1DE& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) override;
        XA1DE& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) override;
        XA1DE& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict(torch::Tensor& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        std::vector<std::vector<double>> predict_proba_threads(const std::vector<std::vector<int>>& test_data);
        std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>>& X) override;
        float score(std::vector<std::vector<int>>& X, std::vector<int>& y) override;
        float score(torch::Tensor& X, torch::Tensor& y) override;
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        int getNumberOfStates() const override;
        int getClassNumStates() const override;
        bayesnet::status_t getStatus() const override { return status; }
        std::string getVersion() override { return version; };
        std::vector<std::string> show() const override { return {}; }
        std::vector<std::string> topological_order()  override { return {}; }
        std::vector<std::string> getNotes() const override { return notes; }
        std::string dump_cpt() const override { return ""; }
        void setHyperparameters(const nlohmann::json& hyperparameters) override;
        std::vector<std::string>& getValidHyperparameters() { return validHyperparameters; }
        void setDebug(bool debug) { this->debug = debug; }
        std::vector<std::string> graph(const std::string& title = "") const override { return {}; }
    protected:
        void trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override {};

    private:
        inline void normalize_weights(int num_instances)
        {
            double sum = std::accumulate(weights_.begin(), weights_.end(), 0.0);
            if (sum == 0) {
                weights_ = std::vector<double>(num_instances, 1.0);
            } else {
                for (double& w : weights_) {
                    w = w * num_instances / sum;
                }
            }
        }
        template <typename T>
        std::vector<T> to_vector(const torch::Tensor& y);
        std::vector<std::vector<int>> to_matrix(const torch::Tensor& X);
        Xaode aode_;
        std::vector<double> weights_;
        CountingSemaphore& semaphore_;
        bool debug = false;
        bayesnet::status_t status = bayesnet::NORMAL;
        std::vector<std::string> notes;
        bool use_threads = false;
        std::string version = "0.9.7";
        bool fitted = false;
    };
}
#endif // XA1DE_H