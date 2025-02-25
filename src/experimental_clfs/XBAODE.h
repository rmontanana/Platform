// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef XBAODE_H
#define XBAODE_H
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include "common/Timer.hpp"
#include "CountingSemaphore.hpp"
#include "bayesnet/ensembles/Boost.h"
#include "XA1DE.h"

namespace platform {
    class XBAODE : public bayesnet::Boost {
    public:
        XBAODE();
        virtual ~XBAODE() = default;
        const std::string CLASSIFIER_NOT_FITTED = "Classifier has not been fitted";
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict(torch::Tensor& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
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
        std::vector<std::string>& getValidHyperparameters() { return validHyperparameters; }
        void setDebug(bool debug) { this->debug = debug; }
        std::vector<std::string> graph(const std::string& title = "") const override { return {}; }
        void set_active_parents(std::vector<int> active_parents) { aode_.set_active_parents(active_parents); }
    protected:
        void trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;
    private:
        std::vector<std::vector<int>> X_train_, X_test_;
        std::vector<int> y_train_, y_test_;
        torch::Tensor dataset;
        XA1DE aode_;
        int n_models;
        std::vector<double> weights_;
        CountingSemaphore& semaphore_;
        bool debug = false;
        bayesnet::status_t status = bayesnet::NORMAL;
        std::vector<std::string> notes;
        bool use_threads = true;
        std::string version = "0.9.7";
        bool fitted = false;
    };
}
#endif // XBAODE_H