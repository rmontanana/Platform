#ifndef HYPERPARAMETERS_H
#define HYPERPARAMETERS_H
#include <string>
#include <map>
#include <vector>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::ordered_json;
    class HyperParameters {
    public:
        HyperParameters() = default;
        // Constructor to use command line hyperparameters
        explicit HyperParameters(const std::vector<std::string>& datasets, const json& hyperparameters_);
        // Constructor to use hyperparameters file generated by grid or by best results
        explicit HyperParameters(const std::vector<std::string>& datasets, const std::string& hyperparameters_file, bool best = false);
        ~HyperParameters() = default;
        bool notEmpty(const std::string& key) const { return !hyperparameters.at(key).empty(); }
        void check(const std::vector<std::string>& valid, const std::string& fileName);
        json get(const std::string& fileName);
    private:
        void normalize_nested(const std::vector<std::string>& datasets);
        std::map<std::string, json> hyperparameters;
        bool best = false; // Used to separate grid/best hyperparameters as the format of those files are different
    };
} /* namespace platform */
#endif