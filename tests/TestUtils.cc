#include "TestUtils.h"
#include "config.h"

class Paths {
public:
    static std::string datasets()
    {
        return { data_path.begin(), data_path.end() };
    }
};

pair<std::vector<mdlp::labels_t>, map<std::string, int>> discretize(std::vector<mdlp::samples_t>& X, mdlp::labels_t& y, std::vector<std::string> features)
{
    std::vector<mdlp::labels_t> Xd;
    map<std::string, int> maxes;
    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        fimdlp.fit(X[i], y);
        mdlp::labels_t& xd = fimdlp.transform(X[i]);
        maxes[features[i]] = *max_element(xd.begin(), xd.end()) + 1;
        Xd.push_back(xd);
    }
    return { Xd, maxes };
}

std::vector<mdlp::labels_t> discretizeDataset(std::vector<mdlp::samples_t>& X, mdlp::labels_t& y)
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

bool file_exists(const std::string& name)
{
    if (FILE* file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::string, map<std::string, std::vector<int>>> loadDataset(const std::string& name, bool class_last, bool discretize_dataset)
{
    auto handler = ArffFiles();
    handler.load(Paths::datasets() + static_cast<std::string>(name) + ".arff", class_last);
    // Get Dataset X, y
    std::vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    std::vector<std::string> features;
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& pair) { return pair.first; });
    torch::Tensor Xd;
    auto states = map<std::string, std::vector<int>>();
    if (discretize_dataset) {
        auto Xr = discretizeDataset(X, y);
        Xd = torch::zeros({ static_cast<int>(Xr.size()), static_cast<int>(Xr[0].size()) }, torch::kInt32);
        for (int i = 0; i < features.size(); ++i) {
            states[features[i]] = std::vector<int>(*max_element(Xr[i].begin(), Xr[i].end()) + 1);
            auto item = states.at(features[i]);
            iota(begin(item), end(item), 0);
            Xd.index_put_({ i, "..." }, torch::tensor(Xr[i], torch::kInt32));
        }
        states[className] = std::vector<int>(*max_element(y.begin(), y.end()) + 1);
        iota(begin(states.at(className)), end(states.at(className)), 0);
    } else {
        Xd = torch::zeros({ static_cast<int>(X.size()), static_cast<int>(X[0].size()) }, torch::kFloat32);
        for (int i = 0; i < features.size(); ++i) {
            Xd.index_put_({ i, "..." }, torch::tensor(X[i]));
        }
    }
    return { Xd, torch::tensor(y, torch::kInt32), features, className, states };
}

tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<std::string>, std::string, map<std::string, std::vector<int>>> loadFile(const std::string& name)
{
    auto handler = ArffFiles();
    handler.load(Paths::datasets() + static_cast<std::string>(name) + ".arff");
    // Get Dataset X, y
    std::vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    std::vector<std::string> features;
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features), [](const auto& pair) { return pair.first; });
    // Discretize Dataset
    std::vector<mdlp::labels_t> Xd;
    map<std::string, int> maxes;
    tie(Xd, maxes) = discretize(X, y, features);
    maxes[className] = *max_element(y.begin(), y.end()) + 1;
    map<std::string, std::vector<int>> states;
    for (auto feature : features) {
        states[feature] = std::vector<int>(maxes[feature]);
    }
    states[className] = std::vector<int>(maxes[className]);
    return { Xd, y, features, className, states };
}
