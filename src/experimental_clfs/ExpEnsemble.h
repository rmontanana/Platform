// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef EXPENSEMBLE_H
#define EXPENSEMBLE_H
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
#include <bayesnet/ensembles/Boost.h>
#include <bayesnet/network/Smoothing.h>
#include "common/Timer.hpp"
#include "CountingSemaphore.hpp"
#include "XSpode.hpp"

namespace platform {
    class ExpEnsemble : public bayesnet::Boost {
    public:
        ExpEnsemble();
        virtual ~ExpEnsemble() = default;
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict(torch::Tensor& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        std::vector<int> predict_spode(std::vector<std::vector<int>>& test_data, int parent);
        std::vector<std::vector<double>> predict_proba(const std::vector<std::vector<int>>& X);
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
    protected:
        void add_model(std::unique_ptr<XSpode> model);
        void remove_last_model();
        bool debug = false;
        std::vector <std::unique_ptr<XSpode>> models_;
        torch::Tensor weights_;
        std::vector<double> significanceModels_;
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
#endif // EXPENSEMBLE_H