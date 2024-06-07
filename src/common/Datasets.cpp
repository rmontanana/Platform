#include <fstream>
#include "Datasets.h"
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::ordered_json;
    const std::string message_dataset_not_loaded = "dataset not loaded.";
    void Datasets::load()
    {
        auto sd = SourceData(sfileType);
        fileType = sd.getFileType();
        path = sd.getPath();
        ifstream catalog(path + "all.txt");
        std::vector<int> numericFeaturesIdx;
        if (!catalog.is_open()) {
            throw std::invalid_argument("Unable to open catalog file. [" + path + "all.txt" + "]");
        }
        std::string line;
        while (getline(catalog, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
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
    std::vector<std::string> Datasets::getNames()
    {
        std::vector<std::string> result;
        transform(datasets.begin(), datasets.end(), back_inserter(result), [](const auto& d) { return d.first; });
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