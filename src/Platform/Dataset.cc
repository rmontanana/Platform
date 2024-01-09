#include "Dataset.h"
#include "ArffFiles.h"
#include <fstream>
namespace platform {
    Dataset::Dataset(const Dataset& dataset) : path(dataset.path), name(dataset.name), className(dataset.className), n_samples(dataset.n_samples), n_features(dataset.n_features), features(dataset.features), states(dataset.states), loaded(dataset.loaded), discretize(dataset.discretize), X(dataset.X), y(dataset.y), Xv(dataset.Xv), Xd(dataset.Xd), yv(dataset.yv), fileType(dataset.fileType)
    {
    }
    std::string Dataset::getName() const
    {
        return name;
    }
    std::string Dataset::getClassName() const
    {
        return className;
    }
    std::vector<std::string> Dataset::getFeatures() const
    {
        if (loaded) {
            return features;
        } else {
            throw std::invalid_argument("Dataset not loaded.");
        }
    }
    int Dataset::getNFeatures() const
    {
        if (loaded) {
            return n_features;
        } else {
            throw std::invalid_argument("Dataset not loaded.");
        }
    }
    int Dataset::getNSamples() const
    {
        if (loaded) {
            return n_samples;
        } else {
            throw std::invalid_argument("Dataset not loaded.");
        }
    }
    std::map<std::string, std::vector<int>> Dataset::getStates() const
    {
        if (loaded) {
            return states;
        } else {
            throw std::invalid_argument("Dataset not loaded.");
        }
    }
    pair<std::vector<std::vector<float>>&, std::vector<int>&> Dataset::getVectors()
    {
        if (loaded) {
            return { Xv, yv };
        } else {
            throw std::invalid_argument("Dataset not loaded.");
        }
    }
    pair<std::vector<std::vector<int>>&, std::vector<int>&> Dataset::getVectorsDiscretized()
    {
        if (loaded) {
            return { Xd, yv };
        } else {
            throw std::invalid_argument("Dataset not loaded.");
        }
    }
    pair<torch::Tensor&, torch::Tensor&> Dataset::getTensors()
    {
        if (loaded) {
            buildTensors();
            return { X, y };
        } else {
            throw std::invalid_argument("Dataset not loaded.");
        }
    }
    void Dataset::load_csv()
    {
        ifstream file(path + "/" + name + ".csv");
        if (file.is_open()) {
            std::string line;
            getline(file, line);
            std::vector<std::string> tokens = split(line, ',');
            features = std::vector<std::string>(tokens.begin(), tokens.end() - 1);
            if (className == "-1") {
                className = tokens.back();
            }
            for (auto i = 0; i < features.size(); ++i) {
                Xv.push_back(std::vector<float>());
            }
            while (getline(file, line)) {
                tokens = split(line, ',');
                for (auto i = 0; i < features.size(); ++i) {
                    Xv[i].push_back(stof(tokens[i]));
                }
                yv.push_back(stoi(tokens.back()));
            }
            file.close();
        } else {
            throw std::invalid_argument("Unable to open dataset file.");
        }
    }
    void Dataset::computeStates()
    {
        for (int i = 0; i < features.size(); ++i) {
            states[features[i]] = std::vector<int>(*max_element(Xd[i].begin(), Xd[i].end()) + 1);
            auto item = states.at(features[i]);
            iota(begin(item), end(item), 0);
        }
        states[className] = std::vector<int>(*max_element(yv.begin(), yv.end()) + 1);
        iota(begin(states.at(className)), end(states.at(className)), 0);
    }
    void Dataset::load_arff()
    {
        auto arff = ArffFiles();
        arff.load(path + "/" + name + ".arff", className);
        // Get Dataset X, y
        Xv = arff.getX();
        yv = arff.getY();
        // Get className & Features
        className = arff.getClassName();
        auto attributes = arff.getAttributes();
        transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& attribute) { return attribute.first; });
    }
    std::vector<std::string> tokenize(std::string line)
    {
        std::vector<std::string> tokens;
        for (auto i = 0; i < line.size(); ++i) {
            if (line[i] == ' ' || line[i] == '\t' || line[i] == '\n') {
                std::string token = line.substr(0, i);
                tokens.push_back(token);
                line.erase(line.begin(), line.begin() + i + 1);
                i = 0;
                while (line[i] == ' ' || line[i] == '\t' || line[i] == '\n')
                    line.erase(line.begin(), line.begin() + i + 1);
            }
        }
        if (line.size() > 0) {
            tokens.push_back(line);
        }
        return tokens;
    }
    void Dataset::load_rdata()
    {
        ifstream file(path + "/" + name + "_R.dat");
        if (file.is_open()) {
            std::string line;
            getline(file, line);
            line = ArffFiles::trim(line);
            std::vector<std::string> tokens = tokenize(line);
            transform(tokens.begin(), tokens.end() - 1, back_inserter(features), [](const auto& attribute) { return ArffFiles::trim(attribute); });
            if (className == "-1") {
                className = ArffFiles::trim(tokens.back());
            }
            for (auto i = 0; i < features.size(); ++i) {
                Xv.push_back(std::vector<float>());
            }
            while (getline(file, line)) {
                tokens = tokenize(line);
                // We have to skip the first token, which is the instance number.
                for (auto i = 1; i < features.size() + 1; ++i) {
                    const float value = stof(tokens[i]);
                    Xv[i - 1].push_back(value);
                }
                yv.push_back(stoi(tokens.back()));
            }
            file.close();
        } else {
            throw std::invalid_argument("Unable to open dataset file.");
        }
    }
    void Dataset::load()
    {
        if (loaded) {
            return;
        }
        if (fileType == CSV) {
            load_csv();
        } else if (fileType == ARFF) {
            load_arff();
        } else if (fileType == RDATA) {
            load_rdata();
        }
        if (discretize) {
            Xd = discretizeDataset(Xv, yv);
            computeStates();
        }
        n_samples = Xv[0].size();
        n_features = Xv.size();
        loaded = true;
    }
    void Dataset::buildTensors()
    {
        if (discretize) {
            X = torch::zeros({ static_cast<int>(n_features), static_cast<int>(n_samples) }, torch::kInt32);
        } else {
            X = torch::zeros({ static_cast<int>(n_features), static_cast<int>(n_samples) }, torch::kFloat32);
        }
        for (int i = 0; i < features.size(); ++i) {
            if (discretize) {
                X.index_put_({ i,  "..." }, torch::tensor(Xd[i], torch::kInt32));
            } else {
                X.index_put_({ i,  "..." }, torch::tensor(Xv[i], torch::kFloat32));
            }
        }
        y = torch::tensor(yv, torch::kInt32);
    }
    std::vector<mdlp::labels_t> Dataset::discretizeDataset(std::vector<mdlp::samples_t>& X, mdlp::labels_t& y)
    {
        std::vector<mdlp::labels_t> Xd;
        auto fimdlp = mdlp::CPPFImdlp();
        for (int i = 0; i < X.size(); i++) {
            fimdlp.fit(X[i], y);
            mdlp::labels_t& xd = fimdlp.transform(X[i]);
            Xd.push_back(xd);
        }
        return Xd;
    }
}