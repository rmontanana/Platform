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
        if (catalog.is_open()) {
            std::string line;
            while (getline(catalog, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                std::vector<std::string> tokens = split(line, ';');
                std::string name = tokens[0];
                std::string className;
                numericFeaturesIdx.clear();
                if (tokens.size() == 1) {
                    className = "-1";
                    numericFeaturesIdx.push_back(-1);
                } else {
                    className = tokens[1];
                    if (tokens.size() > 2) {
                        auto numericFeatures = tokens[2];
                        if (numericFeatures == "all") {
                            numericFeaturesIdx.push_back(-1);
                        } else {
                            auto features = json::parse(numericFeatures);
                            for (auto& f : features) {
                                numericFeaturesIdx.push_back(f);
                            }
                        }
                    } else {
                        numericFeaturesIdx.push_back(-1);
                    }
                }
                datasets[name] = make_unique<Dataset>(path, name, className, discretize, fileType, numericFeaturesIdx);
            }
            catalog.close();
        } else {
            throw std::invalid_argument("Unable to open catalog file. [" + path + "all.txt" + "]");
        }
    }
    std::vector<std::string> Datasets::getNames()
    {
        std::vector<std::string> result;
        transform(datasets.begin(), datasets.end(), back_inserter(result), [](const auto& d) { return d.first; });
        return result;
    }
    std::vector<std::string> Datasets::getFeatures(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getFeatures();
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    std::vector<std::string> Datasets::getLabels(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getLabels();
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    map<std::string, std::vector<int>> Datasets::getStates(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getStates();
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    void Datasets::loadDataset(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return;
        } else {
            datasets.at(name)->load();
        }
    }
    std::string Datasets::getClassName(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getClassName();
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    int Datasets::getNSamples(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getNSamples();
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    int Datasets::getNClasses(const std::string& name)
    {
        if (datasets.at(name)->isLoaded()) {
            auto className = datasets.at(name)->getClassName();
            if (discretize) {
                auto states = getStates(name);
                return states.at(className).size();
            }
            auto [Xv, yv] = getVectors(name);
            return *std::max_element(yv.begin(), yv.end()) + 1;
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    std::vector<bool>& Datasets::getNumericFeatures(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getNumericFeatures();
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    std::vector<int> Datasets::getClassesCounts(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            auto [Xv, yv] = datasets.at(name)->getVectors();
            std::vector<int> counts(*std::max_element(yv.begin(), yv.end()) + 1);
            for (auto y : yv) {
                counts[y]++;
            }
            return counts;
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    pair<std::vector<std::vector<float>>&, std::vector<int>&> Datasets::getVectors(const std::string& name)
    {
        if (!datasets[name]->isLoaded()) {
            datasets[name]->load();
        }
        return datasets[name]->getVectors();
    }
    pair<std::vector<std::vector<int>>&, std::vector<int>&> Datasets::getVectorsDiscretized(const std::string& name)
    {
        if (!datasets[name]->isLoaded()) {
            datasets[name]->load();
        }
        return datasets[name]->getVectorsDiscretized();
    }
    pair<torch::Tensor&, torch::Tensor&> Datasets::getTensors(const std::string& name)
    {
        if (!datasets[name]->isLoaded()) {
            datasets[name]->load();
        }
        return datasets[name]->getTensors();
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