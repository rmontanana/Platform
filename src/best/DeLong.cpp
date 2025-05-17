// DeLong.cpp
// Integración del test de DeLong con la clase RocAuc y Statistics
// Basado en: X. Sun and W. Xu, "Fast Implementation of DeLong’s Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves," (2014), y algoritmos inspirados en sklearn/pROC

#include "DeLong.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cassert>

namespace platform {

    DeLong::DeLongResult DeLong::compare(const std::vector<double>& aucs_model1,
        const std::vector<double>& aucs_model2)
    {
        if (aucs_model1.size() != aucs_model2.size()) {
            throw std::invalid_argument("AUC lists must have the same size");
        }

        size_t N = aucs_model1.size();
        if (N < 2) {
            throw std::invalid_argument("At least two AUC values are required");
        }

        std::vector<double> diffs(N);
        for (size_t i = 0; i < N; ++i) {
            diffs[i] = aucs_model1[i] - aucs_model2[i];
        }

        double mean_diff = std::accumulate(diffs.begin(), diffs.end(), 0.0) / N;
        double var = 0.0;
        for (size_t i = 0; i < N; ++i) {
            var += (diffs[i] - mean_diff) * (diffs[i] - mean_diff);
        }
        var /= (N * (N - 1));
        if (var <= 0.0) var = 1e-10;

        double z = mean_diff / std::sqrt(var);
        double p = 2.0 * (1.0 - std::erfc(std::abs(z) / std::sqrt(2.0)) / 2.0);
        return { mean_diff, z, p };
    }

}
