#ifndef DATASETS_H
#define DATASETS_H
#include "Dataset.h"
namespace platform {
    class Datasets {
    public:
        explicit Datasets(bool discretize, std::string sfileType, std::string discretizer_algorithm = "none");
        std::vector<std::string> getNames();
        bool isDataset(const std::string& name) const;
        Dataset& getDataset(const std::string& name) const { return *datasets.at(name); }
        std::string toString() const;
    private:
        std::string path;
        fileType_t fileType;
        std::string sfileType;
        std::string discretizer_algorithm;
        std::map<std::string, std::unique_ptr<Dataset>> datasets;
        bool discretize;
        void load(); // Loads the list of datasets
    };
};
#endif