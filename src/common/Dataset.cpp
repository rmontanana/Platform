#include <ArffFiles.hpp>
#include <fstream>
#include <set>
#include <nlohmann/json.hpp>
#include "Dataset.h"
namespace platform {
    const std::string message_dataset_not_loaded = "Dataset not loaded.";
    Dataset::Dataset(const Dataset& dataset) :
        path(dataset.path), name(dataset.name), className(dataset.className), n_samples(dataset.n_samples),
        n_features(dataset.n_features), numericFeatures(dataset.numericFeatures), features(dataset.features),
        states(dataset.states), loaded(dataset.loaded), discretize(dataset.discretize), X(dataset.X), y(dataset.y),
        X_train(dataset.X_train), X_test(dataset.X_test), Xv(dataset.Xv), yv(dataset.yv),
        fileType(dataset.fileType)
    {
    }
    std::string Dataset::getName() const
    {
        return name;
    }
    std::vector<std::string> Dataset::getFeatures() const
    {
        if (loaded) {
            return features;
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    long Dataset::getNFeatures() const
    {
        if (loaded) {
            return n_features;
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    long Dataset::getNSamples() const
    {
        if (loaded) {
            return n_samples;
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    std::string Dataset::getClassName() const
    {
        return className;
    }
    int Dataset::getNClasses() const
    {
        if (loaded) {
            return *std::max_element(yv.begin(), yv.end()) + 1;
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    std::vector<std::string> Dataset::getLabels() const
    {
        // Return the labels factorization result
        if (loaded) {
            return labels;
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    std::vector<int> Dataset::getClassesCounts() const
    {
        if (loaded) {
            std::vector<int> counts(*std::max_element(yv.begin(), yv.end()) + 1);
            for (auto y : yv) {
                counts[y]++;
            }
            return counts;
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    std::map<std::string, std::vector<int>> Dataset::getStates() const
    {
        if (loaded) {
            return states;
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    pair<std::vector<std::vector<float>>&, std::vector<int>&> Dataset::getVectors()
    {
        if (loaded) {
            return { Xv, yv };
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    pair<torch::Tensor&, torch::Tensor&> Dataset::getTensors()
    {
        if (loaded) {
            return { X, y };
        } else {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
    }
    void Dataset::load_csv()
    {
        ifstream file(path + "/" + name + ".csv");
        if (!file.is_open()) {
            throw std::invalid_argument("Unable to open dataset file.");
        }
        labels.clear();
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
            auto label = trim(tokens.back());
            if (find(labels.begin(), labels.end(), label) == labels.end()) {
                labels.push_back(label);
            }
            yv.push_back(stoi(label));
        }
        file.close();
    }
    void Dataset::computeStates()
    {
        for (int i = 0; i < features.size(); ++i) {
            auto [max_value, idx] = torch::max(X_train.index({ i, "..." }), 0);
            states[features[i]] = std::vector<int>(max_value.item<int>() + 1);
            iota(begin(states.at(features[i])), end(states.at(features[i])), 0);
        }
        auto [max_value, idx] = torch::max(y_train, 0);
        states[className] = std::vector<int>(max_value.item<int>() + 1);
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
        labels = arff.getLabels();
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
        if (!file.is_open()) {
            throw std::invalid_argument("Unable to open dataset file.");
        }
        std::string line;
        labels.clear();
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
            auto label = trim(tokens.back());
            if (find(labels.begin(), labels.end(), label) == labels.end()) {
                labels.push_back(label);
            }
            yv.push_back(stoi(label));
        }
        file.close();
    }
    void Dataset::load_csv_json()
    {
        using json = nlohmann::ordered_json;
        // 1. Read metadata JSON
        std::ifstream metaFile(path + name + "_metadata.json");
        if (!metaFile.is_open()) {
            throw std::invalid_argument("Unable to open metadata file: " + path + name + "_metadata.json");
        }
        auto metadata = json::parse(metaFile);
        metaFile.close();
        className = metadata.at("target_name");
        // Build set of numeric feature names for quick lookup
        std::set<std::string> numericSet;
        std::set<std::string> categoricalSet;
        if (metadata.contains("feature_types")) {
            auto& ft = metadata["feature_types"];
            if (ft.contains("numeric")) {
                for (auto& f : ft["numeric"]) {
                    numericSet.insert(f.get<std::string>());
                }
            }
            if (ft.contains("categorical")) {
                for (auto& f : ft["categorical"]) {
                    categoricalSet.insert(f.get<std::string>());
                }
            }
        }
        // 2. Read CSV
        std::ifstream file(path + name + ".csv");
        if (!file.is_open()) {
            throw std::invalid_argument("Unable to open dataset file: " + path + name + ".csv");
        }
        labels.clear();
        std::string line;
        // Read header
        getline(file, line);
        std::vector<std::string> header = split(line, ',');
        // All columns except the last are features (target is last column)
        features = std::vector<std::string>(header.begin(), header.end() - 1);
        for (auto i = 0; i < features.size(); ++i) {
            features[i] = trim(features[i]);
            Xv.push_back(std::vector<float>());
        }
        // Factorization maps for categorical features (per feature index)
        std::vector<std::map<std::string, int>> factorMaps(features.size());
        // Factorization map for target
        std::map<std::string, int> labelMap;
        // Read data rows, skipping rows with missing data
        while (getline(file, line)) {
            if (line.empty()) continue;
            std::vector<std::string> tokens = split(line, ',');
            // Check for missing values in any column
            bool has_missing = false;
            for (auto i = 0; i < tokens.size(); ++i) {
                if (trim(tokens[i]).empty()) {
                    has_missing = true;
                    break;
                }
            }
            if (has_missing) continue;
            for (auto i = 0; i < features.size(); ++i) {
                auto value = trim(tokens[i]);
                if (numericSet.count(features[i])) {
                    // Numeric feature
                    Xv[i].push_back(stof(value));
                } else {
                    // Categorical feature → factorize
                    if (factorMaps[i].find(value) == factorMaps[i].end()) {
                        factorMaps[i][value] = static_cast<int>(factorMaps[i].size());
                    }
                    Xv[i].push_back(static_cast<float>(factorMaps[i][value]));
                }
            }
            // Target (last column) → factorize
            auto label = trim(tokens.back());
            if (labelMap.find(label) == labelMap.end()) {
                labelMap[label] = static_cast<int>(labelMap.size());
                labels.push_back(label);
            }
            yv.push_back(labelMap[label]);
        }
        file.close();
        // 3. Build numericFeatures[] from metadata
        numericFeatures.resize(features.size(), false);
        numericFeaturesIdx.clear();
        for (int i = 0; i < features.size(); ++i) {
            if (numericSet.count(features[i])) {
                numericFeatures[i] = true;
                numericFeaturesIdx.push_back(i);
            }
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
        } else if (fileType == CSVJSON) {
            load_csv_json();
        }
        n_samples = Xv[0].size();
        n_features = Xv.size();
        if (fileType != CSVJSON) {
            // CSVJSON builds numericFeatures inside load_csv_json()
            if (numericFeaturesIdx.size() == 0) {
                // None of the features are numeric
                numericFeatures = std::vector<bool>(n_features, false);
            } else {
                if (numericFeaturesIdx.at(0) == -1) {
                    // All the features are numeric
                    numericFeatures = std::vector<bool>(n_features, true);
                } else {
                    // Only some of the features are numeric
                    numericFeatures = std::vector<bool>(n_features, false);
                    for (auto i : numericFeaturesIdx) {
                        numericFeatures[i] = true;
                    }
                }
            }
        }
        // Build Tensors
        X = torch::zeros({ n_features, n_samples }, torch::kFloat32);
        for (int i = 0; i < features.size(); ++i) {
            X.index_put_({ i,  "..." }, torch::tensor(Xv[i], torch::kFloat32));
        }
        y = torch::tensor(yv, torch::kInt32);
        loaded = true;
    }
    std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&> Dataset::getTrainTestTensors(std::vector<int>& train, std::vector<int>& test)
    {
        if (!loaded) {
            throw std::invalid_argument(message_dataset_not_loaded);
        }
        auto train_t = torch::tensor(train);
        int samples_train = train.size();
        int samples_test = test.size();
        auto test_t = torch::tensor(test);
        X_train = X.index({ "...", train_t });
        y_train = y.index({ train_t });
        X_test = X.index({ "...", test_t });
        y_test = y.index({ test_t });
        if (discretize) {
            auto discretizer = Discretization::instance()->create(discretizer_algorithm);
            auto X_train_d = torch::zeros({ n_features, samples_train }, torch::kInt32);
            auto X_test_d = torch::zeros({ n_features, samples_test }, torch::kInt32);
            for (auto feature = 0; feature < n_features; ++feature) {
                if (numericFeatures[feature]) {
                    auto feature_train = X_train.index({ feature, "..." });
                    auto feature_test = X_test.index({ feature, "..." });
                    auto feature_train_disc = discretizer->fit_transform_t(feature_train, y_train);
                    auto feature_test_disc = discretizer->transform_t(feature_test);
                    X_train_d.index_put_({ feature, "..." }, feature_train_disc);
                    X_test_d.index_put_({ feature, "..." }, feature_test_disc);
                } else {
                    X_train_d.index_put_({ feature, "..." }, X_train.index({ feature, "..." }).to(torch::kInt32));
                    X_test_d.index_put_({ feature, "..." }, X_test.index({ feature, "..." }).to(torch::kInt32));
                }
            }
            X_train = X_train_d;
            X_test = X_test_d;
            assert(X_train.dtype() == torch::kInt32);
            assert(X_test.dtype() == torch::kInt32);
            computeStates();
        }
        assert(y_train.dtype() == torch::kInt32);
        assert(y_test.dtype() == torch::kInt32);
        return { X_train, X_test, y_train, y_test };
    }
}