#ifndef ROCAUC_H
#define ROCAUC_H
#include <torch/torch.h>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::ordered_json;
    class RocAuc {
    public:
        RocAuc() = default;
        double compute(const std::vector<std::vector<double>>& y_proba, const std::vector<int>& y_test);
        double compute(const torch::Tensor& y_proba, const torch::Tensor& y_test);
    private:
        double compute_common(size_t nSamples, size_t classIdx);
        std::vector<std::pair<double, int>> scoresAndLabels;
        std::vector<int> y_test;
    };
}
#endif