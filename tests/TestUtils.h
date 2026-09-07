#ifndef TESTUTILS_H
#define TESTUTILS_H
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <ArffFiles/ArffFiles.hpp>
#include <fimdlp/CPPFImdlp.h>

bool file_exists(const std::string& name);
std::pair<std::vector<mdlp::labels_t>, std::map<std::string, int>> discretize(std::vector<mdlp::samples_t>& X, mdlp::labels_t& y, std::vector<std::string> features);
std::vector<mdlp::labels_t> discretizeDataset(std::vector<mdlp::samples_t>& X, mdlp::labels_t& y);
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<std::string>, std::string, std::map<std::string, std::vector<int>>> loadFile(const std::string& name);
std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::string, std::map<std::string, std::vector<int>>> loadDataset(const std::string& name, bool class_last, bool discretize_dataset);

class RawDatasets {
public:
    RawDatasets(const std::string& file_name, bool discretize)
    {
        // Xt can be either discretized or not
        std::tie(Xt, yt, featurest, classNamet, statest) = loadDataset(file_name, true, discretize);
        // Xv is always discretized
        std::tie(Xv, yv, featuresv, classNamev, statesv) = loadFile(file_name);
        auto yresized = torch::transpose(yt.view({ yt.size(0), 1 }), 0, 1);
        dataset = torch::cat({ Xt, yresized }, 0);
        nSamples = dataset.size(1);
        weights = torch::full({ nSamples }, 1.0 / nSamples, torch::kDouble);
        weightsv = std::vector<double>(nSamples, 1.0 / nSamples);
        classNumStates = discretize ? statest.at(classNamet).size() : 0;
    }
    torch::Tensor Xt, yt, dataset, weights;
    std::vector<std::vector<int>> Xv;
    std::vector<double> weightsv;
    std::vector<int> yv;
    std::vector<std::string> featurest, featuresv;
    std::map<std::string, std::vector<int>> statest, statesv;
    std::string classNamet, classNamev;
    int nSamples, classNumStates;
    double epsilon = 1e-5;
};
#endif