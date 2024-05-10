#pragma once

#include <torch/torch.h>
#include <map>
#include <vector>
#include <string>
#include <CPPFImdlp.h>
#include "Utils.h"
#include "SourceData.h"
namespace platform {

    class Dataset {
    public:
        Dataset(const std::string& path, const std::string& name, const std::string& className, bool discretize, fileType_t fileType) : path(path), name(name), className(className), discretize(discretize), loaded(false), fileType(fileType) {};
        explicit Dataset(const Dataset&);
        std::string getName() const;
        std::string getClassName() const;
        std::vector<std::string> getLabels() const { return labels; }
        std::vector<string> getFeatures() const;
        std::map<std::string, std::vector<int>> getStates() const;
        std::pair<vector<std::vector<float>>&, std::vector<int>&> getVectors();
        std::pair<vector<std::vector<int>>&, std::vector<int>&> getVectorsDiscretized();
        std::pair<torch::Tensor&, torch::Tensor&> getTensors();
        int getNFeatures() const;
        int getNSamples() const;
        void load();
        const bool inline isLoaded() const { return loaded; };
    private:
        std::string path;
        std::string name;
        fileType_t fileType;
        std::string className;
        int n_samples{ 0 }, n_features{ 0 };
        std::vector<std::string> features;
        std::vector<std::string> labels;
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
    };
};

