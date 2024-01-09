#include "Datasets.h"
#include <fstream>
namespace platform {
    void Datasets::load()
    {
        auto sd = SourceData(sfileType);
        fileType = sd.getFileType();
        path = sd.getPath();
        ifstream catalog(path + "all.txt");
        if (catalog.is_open()) {
            std::string line;
            while (getline(catalog, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                std::vector<std::string> tokens = split(line, ',');
                std::string name = tokens[0];
                std::string className;
                if (tokens.size() == 1) {
                    className = "-1";
                } else {
                    className = tokens[1];
                }
                datasets[name] = make_unique<Dataset>(path, name, className, discretize, fileType);
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
            throw std::invalid_argument("Dataset not loaded.");
        }
    }
    map<std::string, std::vector<int>> Datasets::getStates(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getStates();
        } else {
            throw std::invalid_argument("Dataset not loaded.");
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
            throw std::invalid_argument("Dataset not loaded.");
        }
    }
    int Datasets::getNSamples(const std::string& name) const
    {
        if (datasets.at(name)->isLoaded()) {
            return datasets.at(name)->getNSamples();
        } else {
            throw std::invalid_argument("Dataset not loaded.");
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
            throw std::invalid_argument("Dataset not loaded.");
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
            throw std::invalid_argument("Dataset not loaded.");
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
}