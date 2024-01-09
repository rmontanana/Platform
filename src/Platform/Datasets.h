#ifndef DATASETS_H
#define DATASETS_H
#include "Dataset.h"
namespace platform {
    class Datasets {
    private:
        std::string path;
        fileType_t fileType;
        std::string sfileType;
        std::map<std::string, std::unique_ptr<Dataset>> datasets;
        bool discretize;
        void load(); // Loads the list of datasets
    public:
        explicit Datasets(bool discretize, std::string sfileType) : discretize(discretize), sfileType(sfileType) { load(); };
        std::vector<string> getNames();
        std::vector<string> getFeatures(const std::string& name) const;
        int getNSamples(const std::string& name) const;
        std::string getClassName(const std::string& name) const;
        int getNClasses(const std::string& name);
        std::vector<int> getClassesCounts(const std::string& name) const;
        std::map<std::string, std::vector<int>> getStates(const std::string& name) const;
        std::pair<std::vector<std::vector<float>>&, std::vector<int>&> getVectors(const std::string& name);
        std::pair<std::vector<std::vector<int>>&, std::vector<int>&> getVectorsDiscretized(const std::string& name);
        std::pair<torch::Tensor&, torch::Tensor&> getTensors(const std::string& name);
        bool isDataset(const std::string& name) const;
        void loadDataset(const std::string& name) const;
    };
};

#endif