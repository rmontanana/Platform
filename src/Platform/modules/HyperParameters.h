#ifndef HYPERPARAMETERS_H
#define HYPERPARAMETERS_H
#include <string>
#include <map>
#include <vector>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::json;
    class HyperParameters {
    public:
        HyperParameters() = default;
        explicit HyperParameters(const std::vector<std::string>& datasets, const json& hyperparameters_);
        explicit HyperParameters(const std::vector<std::string>& datasets, const std::string& hyperparameters_file);
        ~HyperParameters() = default;
        bool notEmpty(const std::string& key) const { return !hyperparameters.at(key).empty(); }
        void check(const std::vector<std::string>& valid, const std::string& fileName);
        json get(const std::string& fileName);
    private:
        std::map<std::string, json> hyperparameters;
    };
} /* namespace platform */
#endif /* HYPERPARAMETERS_H */