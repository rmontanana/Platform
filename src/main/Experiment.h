#pragma once

#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <string>
#include <folding.hpp>
#include "bayesnet/BaseClassifier.h"
#include "HyperParameters.h"
#include "results/Result.h"

namespace platform {
    using json = nlohmann::json;

    class Experiment {
    public:
        Experiment() = default;
        Experiment& setPlatform(const std::string& platform) { this->result.setPlatform(platform); return *this; }
        Experiment& setScoreName(const std::string& score_name) { this->result.setScoreName(score_name); return *this; }
        Experiment& setTitle(const std::string& title) { this->result.setTitle(title); return *this; }
        Experiment& setModelVersion(const std::string& model_version) { this->result.setModelVersion(model_version); return *this; }
        Experiment& setModel(const std::string& model) { this->result.setModel(model); return *this; }
        Experiment& setLanguage(const std::string& language) { this->result.setLanguage(language); return *this; }
        Experiment& setLanguageVersion(const std::string& language_version) { this->result.setLanguageVersion(language_version); return *this; }
        Experiment& setDiscretized(bool discretized) { this->discretized = discretized; result.setDiscretized(discretized); return *this; }
        Experiment& setStratified(bool stratified) { this->stratified = stratified; result.setStratified(stratified); return *this; }
        Experiment& setNFolds(int nfolds) { this->nfolds = nfolds; result.setNFolds(nfolds); return *this; }
        Experiment& addResult(PartialResult result_) { result.addPartial(result_); return *this; }
        Experiment& addRandomSeed(int randomSeed) { randomSeeds.push_back(randomSeed); result.addSeed(randomSeed); return *this; }
        Experiment& setDuration(float duration) { this->result.setDuration(duration); return *this; }
        Experiment& setHyperparameters(const HyperParameters& hyperparameters_) { this->hyperparameters = hyperparameters_; return *this; }
        void cross_validation(const std::string& fileName, bool quiet, bool no_train_score);
        void go(std::vector<std::string> filesToProcess, bool quiet, bool no_train_score);
        void saveResult();
        void show();
        void report();
    private:
        Result result;
        bool discretized{ false }, stratified{ false };
        std::vector<PartialResult> results;
        std::vector<int> randomSeeds;
        HyperParameters hyperparameters;
        int nfolds{ 0 };
        int max_name{ 7 }; // max length of dataset name for formatting (default 7)
    };
}