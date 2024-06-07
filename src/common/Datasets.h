#ifndef DATASETS_H
#define DATASETS_H
#include "Dataset.h"
namespace platform {
    class Datasets {
    public:
        explicit Datasets(bool discretize, std::string sfileType, std::string discretizer_algo = "none") : discretize(discretize), sfileType(sfileType), discretizer_algo(discretizer_algo)
        {
            if (discretizer_algo == "none" && discretize) {
                throw std::runtime_error("Can't discretize without discretization algorithm");
            }
            load();
        };
        std::vector<std::string> getNames();
        std::vector<std::string> getFeatures(const std::string& name) const;
        int getNSamples(const std::string& name) const;
        std::vector<std::string> getLabels(const std::string& name) const;
        std::string getClassName(const std::string& name) const;
        int getNClasses(const std::string& name);
        std::vector<bool>& getNumericFeatures(const std::string& name) const;
        std::vector<int> getClassesCounts(const std::string& name) const;
        std::map<std::string, std::vector<int>> getStates(const std::string& name) const;
        std::pair<std::vector<std::vector<float>>&, std::vector<int>&> getVectors(const std::string& name);
        std::pair<std::vector<std::vector<int>>&, std::vector<int>&> getVectorsDiscretized(const std::string& name);
        std::pair<torch::Tensor&, torch::Tensor&> getTensors(const std::string& name);
        std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&> getTrainTestTensors(const std::vector<int>& train_idx, const std::vector<int>& test_idx);
        bool isDataset(const std::string& name) const;
        void loadDataset(const std::string& name) const;
        std::string toString() const;
    private:
        std::string path;
        fileType_t fileType;
        std::string sfileType;
        std::string discretizer_algo;
        std::map<std::string, std::unique_ptr<Dataset>> datasets;
        bool discretize;
        void load(); // Loads the list of datasets
    };
};
#endif