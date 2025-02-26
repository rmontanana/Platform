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
#include "common/Timer.hpp"
#include "Xaode.hpp"
#include "ExpClf.h"

namespace platform {
    class XA1DE : public ExpClf {
    public:
        XA1DE() = default;
        virtual ~XA1DE() = default;
        XA1DE& fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) override;
        XA1DE& fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) override;
        XA1DE& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing) override;
        XA1DE& fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;
        std::string getVersion() override { return version; };
    protected:
        void trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override {};
    private:
        std::string version = "1.0.0";
    };
}
#endif // XA1DE_H