#include <fstream>
#include <algorithm>
#include <filesystem>
#include "Datasets.h"
#include "DotEnv.h"
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::ordered_json;
    const std::string message_dataset_not_loaded = "dataset not loaded.";
    Datasets::Datasets(bool discretize, std::string sfileType, std::string discretizer_algorithm) :
        discretize(discretize), sfileType(sfileType), discretizer_algorithm(discretizer_algorithm)
    {
        if ((discretizer_algorithm == "none" || discretizer_algorithm == "") && discretize) {
            throw std::runtime_error("Can't discretize without discretization algorithm");
        }
        load();
    }
    void Datasets::load()
    {
        auto sd = SourceData(sfileType);
        fileType = sd.getFileType();
        path = sd.getPath();
        if (fileType == CSVJSON) {
            loadCsvJson();
            return;
        }
        ifstream catalog(path + "all.txt");
        std::vector<int> numericFeaturesIdx;
        if (!catalog.is_open()) {
            throw std::invalid_argument("Unable to open catalog file. [" + path + "all.txt" + "]");
        }
        std::string line;
        std::vector<std::string> sorted_lines;
        while (getline(catalog, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            sorted_lines.push_back(line);
        }
        sort(sorted_lines.begin(), sorted_lines.end(), [](const auto& lhs, const auto& rhs) {
            const auto result = mismatch(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(), [](const auto& lhs, const auto& rhs) {return tolower(lhs) == tolower(rhs);});

            return result.second != rhs.cend() && (result.first == lhs.cend() || tolower(*result.first) < tolower(*result.second));
            });

        for (const auto& line : sorted_lines) {
            std::vector<std::string> tokens = split(line, ';');
            std::string name = tokens[0];
            std::string className;
            numericFeaturesIdx.clear();
            int size = tokens.size();
            switch (size) {
                case 1:
                    className = "-1";
                    numericFeaturesIdx.push_back(-1);
                    break;
                case 2:
                    className = tokens[1];
                    numericFeaturesIdx.push_back(-1);
                    break;
                case 3:
                    {
                        className = tokens[1];
                        auto numericFeatures = tokens[2];
                        if (numericFeatures == "all") {
                            numericFeaturesIdx.push_back(-1);
                        } else {
                            if (numericFeatures != "none") {
                                auto features = json::parse(numericFeatures);
                                for (auto& f : features) {
                                    numericFeaturesIdx.push_back(f);
                                }
                            }
                        }
                    }
                    break;
                default:
                    throw std::invalid_argument("Invalid catalog file format.");

            }
            datasets[name] = make_unique<Dataset>(path, name, className, discretize, fileType, numericFeaturesIdx, discretizer_algorithm);
        }
        catalog.close();
    }
    void Datasets::loadCsvJson()
    {
        auto env = DotEnv();
        path = env.get("csv_json_path");
        if (path.empty()) {
            throw std::invalid_argument("csv_json_path is required in .env when source_data=CsvJSON");
        }
        if (!path.empty() && path.back() != '/') {
            path += '/';
        }
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.path().extension() != ".json") {
                continue;
            }
            auto stem = entry.path().stem().string();
            const std::string suffix = "_metadata";
            if (stem.size() <= suffix.size() || stem.substr(stem.size() - suffix.size()) != suffix) {
                continue;
            }
            auto name = stem.substr(0, stem.size() - suffix.size());
            // Read metadata JSON to get className and feature types
            std::ifstream metaFile(entry.path());
            if (!metaFile.is_open()) {
                throw std::invalid_argument("Unable to open metadata file: " + entry.path().string());
            }
            auto metadata = json::parse(metaFile);
            metaFile.close();
            std::string className = metadata.at("target_name");
            // numericFeaturesIdx is not used for CSVJSON - load_csv_json() resolves
            // numeric features by name from the metadata JSON directly
            std::vector<int> numericFeaturesIdx;
            datasets[name] = make_unique<Dataset>(path, name, className, discretize, fileType, numericFeaturesIdx, discretizer_algorithm);
        }
    }
    std::vector<std::string> Datasets::getNames()
    {
        std::vector<std::string> result;
        transform(datasets.begin(), datasets.end(), back_inserter(result), [](const auto& d) { return d.first; });
        sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
            const auto result = mismatch(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(), [](const auto& lhs, const auto& rhs) {return tolower(lhs) == tolower(rhs);});

            return result.second != rhs.cend() && (result.first == lhs.cend() || tolower(*result.first) < tolower(*result.second));
            });
        return result;
    }
    bool Datasets::isDataset(const std::string& name) const
    {
        return datasets.find(name) != datasets.end();
    }
    std::string Datasets::toString() const
    {
        std::string result;
        std::string sep = "";
        for (const auto& d : datasets) {
            result += sep + d.first;
            sep = ", ";
        }
        return "{" + result + "}";
    }
}