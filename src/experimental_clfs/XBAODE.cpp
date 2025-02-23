// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "XBAODE.h"

namespace platform {
    XBAODE::XBAODE() : semaphore_{ CountingSemaphore::getInstance() }
    {
    }
    XBAODE& XBAODE::fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        aode_.fit(X, y, features, className, states, smoothing);
        fitted = true;
        return *this;
    }
    std::vector<std::vector<double>> XBAODE::predict_proba(std::vector<std::vector<int>>& test_data)
    {
        return aode_.predict_proba_threads(test_data);
    }
    std::vector<int> XBAODE::predict(std::vector<std::vector<int>>& test_data)
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        return aode_.predict(test_data);
    }
    float XBAODE::score(std::vector<std::vector<int>>& test_data, std::vector<int>& labels)
    {
        return aode_.score(test_data, labels);
    }

    //
    // statistics
    //
    int XBAODE::getNumberOfNodes() const
    {
        return aode_.getNumberOfNodes();
    }
    int XBAODE::getNumberOfEdges() const
    {
        return aode_.getNumberOfEdges();
    }
    int XBAODE::getNumberOfStates() const
    {
        return aode_.getNumberOfStates();
    }
    int XBAODE::getClassNumStates() const
    {
        return aode_.getClassNumStates();
    }

    //
    // Fit
    //
    // fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    XBAODE& XBAODE::fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        aode_.fit(X, y, features, className, states, smoothing);
        return *this;
    }
    XBAODE& XBAODE::fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        aode_.fit(dataset, features, className, states, smoothing);
        return *this;
    }
    XBAODE& XBAODE::fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing)
    {
        aode_.fit(dataset, features, className, states, weights, smoothing);
        return *this;
    }
    //
    // Predict
    //
    torch::Tensor XBAODE::predict(torch::Tensor& X)
    {
        return aode_.predict(X);
    }
    torch::Tensor XBAODE::predict_proba(torch::Tensor& X)
    {
        return aode_.predict_proba(X);
    }
    float XBAODE::score(torch::Tensor& X, torch::Tensor& y)
    {
        return aode_.score(X, y);
    }

}