#include <fstream>
#include <sstream>
#include <iostream>
#include "HyperParameters.h"

namespace platform {
    HyperParameters::HyperParameters(const std::vector<std::string>& datasets, const json& hyperparameters_)
    {
        // Initialize all datasets with the given hyperparameters
        for (const auto& item : datasets) {
            hyperparameters[item] = hyperparameters_;
        }
        normalize_nested(datasets);
    }
    HyperParameters::HyperParameters(const std::vector<std::string>& datasets, const std::string& hyperparameters_file, bool best)
    {
        // Check if file exists
        std::ifstream file(hyperparameters_file);
        if (!file.is_open()) {
            throw std::runtime_error("File " + hyperparameters_file + " not found");
        }
        // Check if file is a json
        json file_hyperparameters = json::parse(file);
        json input_hyperparameters;
        if (best) {
            for (const auto& [key, value] : file_hyperparameters.items()) {
                input_hyperparameters[key]["hyperparameters"] = value[1];
            }
        } else {
            input_hyperparameters = file_hyperparameters["results"];
        }
        // Check if hyperparameters are valid
        for (const auto& dataset : datasets) {
            if (!input_hyperparameters.contains(dataset)) {
                std::cerr << "*Warning: Dataset " << dataset << " not found in hyperparameters file" << " assuming default hyperparameters" << std::endl;
                hyperparameters[dataset] = json({});
                continue;
            }
            hyperparameters[dataset] = input_hyperparameters[dataset]["hyperparameters"].get<json>();
        }
        normalize_nested(datasets);
    }
    void HyperParameters::normalize_nested(const std::vector<std::string>& datasets)
    {
        for (const auto& dataset : datasets) {
            if (hyperparameters[dataset].contains("be_hyperparams")) {
                // Odte has base estimator hyperparameters set this way
                hyperparameters[dataset]["be_hyperparams"] = hyperparameters[dataset]["be_hyperparams"].dump();
            }
        }
    }
    // https://www.techiedelight.com/implode-a-vector-of-strings-into-a-comma-separated-string-in-cpp/
    std::string join(std::vector<std::string> const& strings, std::string delim)
    {
        std::stringstream ss;
        std::copy(strings.begin(), strings.end(),
            std::ostream_iterator<std::string>(ss, delim.c_str()));
        return ss.str();
    }
    void HyperParameters::check(const std::vector<std::string>& valid, const std::string& fileName)
    {
        json result = hyperparameters.at(fileName);
        for (const auto& item : result.items()) {
            if (find(valid.begin(), valid.end(), item.key()) == valid.end()) {
                throw std::invalid_argument("Hyperparameter " + item.key() + " is not valid. Passed Hyperparameters are: "
                    + result.dump(4) + "\n Valid hyperparameters are: {" + join(valid, ",") + "}");
            }
        }
    }
    json HyperParameters::get(const std::string& fileName)
    {
        return hyperparameters.at(fileName);
    }
} /* namespace platform */