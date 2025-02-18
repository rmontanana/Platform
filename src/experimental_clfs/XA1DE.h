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
        void setDebug(bool debug) { this->debug = debug; }
        std::vector<std::vector<double>> predict_proba_threads(const std::vector<std::vector<int>>& test_data);

        XA1DE& fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) override;
        XA1DE& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) override;
        XA1DE& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) override;
        XA1DE& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;
        int getNumberOfNodes() const override { return 0; };
        int getNumberOfEdges() const override { return 0; };
        int getNumberOfStates() const override { return 0; };
        int getClassNumStates() const override { return 0; };
        torch::Tensor predict(torch::Tensor& X) override { return torch::zeros(0); };
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override { return torch::zeros(0); };
        std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>>& X) override;
        bayesnet::status_t getStatus() const override { return status; }
        std::string getVersion() override { return { project_version.begin(), project_version.end() }; };
        float score(torch::Tensor& X, torch::Tensor& y) override { return 0; };
        float score(std::vector<std::vector<int>>& X, std::vector<int>& y) override;
        std::vector<std::string> show() const override { return {}; }
        std::vector<std::string> topological_order()  override { return {}; }
        std::vector<std::string> getNotes() const override { return notes; }
        std::string dump_cpt() const override { return ""; }
        void setHyperparameters(const nlohmann::json& hyperparameters) override;

        std::vector<std::string>& getValidHyperparameters() { return validHyperparameters; }
    protected:
        void trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;

    private:
        inline void normalize_weights(int num_instances)
        {
            double sum = std::accumulate(weights_.begin(), weights_.end(), 0.0);
            if (sum == 0) {
                throw std::runtime_error("Weights sum zero.");
            }
            for (double& w : weights_) {
                w = w * num_instances / sum;
            }
        }
        // The instances of the dataset
        Xaode aode_;
        std::vector<double> weights_;
        CountingSemaphore& semaphore_;
        bool debug = false;
        bayesnet::status_t status = bayesnet::NORMAL;
        std::vector<std::string> notes;
        bool use_threads = false;
    };
}
#endif // XA1DE_H