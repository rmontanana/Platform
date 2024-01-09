#ifndef DATASET_H
#define DATASET_H
#include <torch/torch.h>
#include <map>
#include <vector>
#include <string>
#include "CPPFImdlp.h"
#include "Utils.h"
namespace platform {
    enum fileType_t { CSV, ARFF, RDATA };
    class SourceData {
    public:
        SourceData(std::string source)
        {
            if (source == "Surcov") {
                path = "datasets/";
                fileType = CSV;
            } else if (source == "Arff") {
                path = "datasets/";
                fileType = ARFF;
            } else if (source == "Tanveer") {
                path = "data/";
                fileType = RDATA;
            } else {
                throw std::invalid_argument("Unknown source.");
            }
        }
        std::string getPath()
        {
            return path;
        }
        fileType_t getFileType()
        {
            return fileType;
        }
    private:
        std::string path;
        fileType_t fileType;
    };
    class Dataset {
    private:
        std::string path;
        std::string name;
        fileType_t fileType;
        std::string className;
        int n_samples{ 0 }, n_features{ 0 };
        std::vector<std::string> features;
        std::map<std::string, std::vector<int>> states;
        bool loaded;
        bool discretize;
        torch::Tensor X, y;
        std::vector<std::vector<float>> Xv;
        std::vector<std::vector<int>> Xd;
        std::vector<int> yv;
        void buildTensors();
        void load_csv();
        void load_arff();
        void load_rdata();
        void computeStates();
        std::vector<mdlp::labels_t> discretizeDataset(std::vector<mdlp::samples_t>& X, mdlp::labels_t& y);
    public:
        Dataset(const std::string& path, const std::string& name, const std::string& className, bool discretize, fileType_t fileType) : path(path), name(name), className(className), discretize(discretize), loaded(false), fileType(fileType) {};
        explicit Dataset(const Dataset&);
        std::string getName() const;
        std::string getClassName() const;
        std::vector<string> getFeatures() const;
        std::map<std::string, std::vector<int>> getStates() const;
        std::pair<vector<std::vector<float>>&, std::vector<int>&> getVectors();
        std::pair<vector<std::vector<int>>&, std::vector<int>&> getVectorsDiscretized();
        std::pair<torch::Tensor&, torch::Tensor&> getTensors();
        int getNFeatures() const;
        int getNSamples() const;
        void load();
        const bool inline isLoaded() const { return loaded; };
    };
};

#endif