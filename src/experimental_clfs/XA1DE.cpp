// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "XA1DE.h"
#include "TensorUtils.hpp"

namespace platform {
    XA1DE& XA1DE::fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        std::vector<std::vector<int>> instances = X;
        instances.push_back(y);
        int num_instances = instances[0].size();
        int num_attributes = instances.size();
        normalize_weights(num_instances);
        aode_.fit(X, y, features, className, states, weights_, true);
        fitted = true;
        return *this;
    }
    //
    // Fit
    //
    XA1DE& XA1DE::fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto y_ = TensorUtils::to_vector<int>(y);
        return fit(X_, y_, features, className, states, smoothing);
    }
    XA1DE& XA1DE::fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        torch::Tensor y = dataset[dataset.size(0) - 1];
        torch::Tensor X = dataset.slice(0, 0, dataset.size(0) - 1);
        return fit(X, y, features, className, states, smoothing);
    }
    XA1DE& XA1DE::fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing)
    {
        weights_ = weights;
        return fit(dataset, features, className, states, smoothing);
    }
}