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
#include "ExpClf.h"

namespace platform {
    class XBAODE {

        // Hay que hacer un vector de modelos entrenados y hacer un predict ensemble con todos ellos
        // Probar XA1DE con smooth original y laplace y comprobar diferencias si se pasan pesos a 1 o a 1/m
    public:
        XBAODE();
        std::string getVersion() override { return version; };
    protected:
        void trainModel(const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing) override;
    private:
        std::vector<std::vector<int>> X_train_, X_test_;
        std::vector<int> y_train_, y_test_;
        torch::Tensor dataset;
        int n_models;
        std::string version = "0.9.7";
    };
}
#endif // XBAODE_H