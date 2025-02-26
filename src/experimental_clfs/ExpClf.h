// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef EXPCLF_H
#define EXPCLF_H
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
#include "bayesnet/BaseClassifier.h"
#include "common/Timer.hpp"
#include "CountingSemaphore.hpp"
#include "Xaode.hpp"

namespace platform {

    class ExpClf : public bayesnet::BaseClassifier {
    public:
        ExpClf();
        virtual ~ExpClf() = default;
        ExpClf& fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) { return *this; };
        // X is nxm tensor, y is nx1 tensor
        ExpClf& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) { return *this; };
        ExpClf& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) { return *this; };
        ExpClf& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) { return *this; };
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict(torch::Tensor& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        std::vector<int> predict_spode(std::vector<std::vector<int>>& test_data, int parent);
        std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>>& X) override;
        float score(std::vector<std::vector<int>>& X, std::vector<int>& y) override;
        float score(torch::Tensor& X, torch::Tensor& y) override;
        int getNumberOfNodes() const override;
        int getNumberOfEdges() const override;
        int getNumberOfStates() const override;
        int getClassNumStates() const override;
        std::vector<std::string> show() const override { return {}; }
        std::vector<std::string> topological_order()  override { return {}; }
        std::string dump_cpt() const override { return ""; }
        void setDebug(bool debug) { this->debug = debug; }
        bayesnet::status_t getStatus() const override { return status; }
        std::vector<std::string> getNotes() const override { return notes; }
        std::vector<std::string> graph(const std::string& title = "") const override { return {}; }
        void setHyperparameters(const nlohmann::json& hyperparameters) override;
        void set_active_parents(std::vector<int> active_parents) { for (const auto& parent : active_parents) aode_.add_active_parent(parent); }
        void add_active_parent(int parent) { aode_.add_active_parent(parent); }
        void remove_last_parent() { aode_.remove_last_parent(); }
    protected:
        bool debug = false;
        Xaode aode_;
        torch::Tensor weights_;
        bool fitted = false;
        const std::string CLASSIFIER_NOT_FITTED = "Classifier has not been fitted";
        inline void normalize_weights(int num_instances)
        {
            double sum = weights_.sum().item<double>();
            if (sum == 0) {
                weights_ = torch::full({ num_instances }, 1.0);
            } else {
                for (int i = 0; i < weights_.size(0); ++i) {
                    weights_[i] = weights_[i].item<double>() * num_instances / sum;
                }
            }
        }
    private:
        CountingSemaphore& semaphore_;
    };
}
#endif // EXPCLF_H