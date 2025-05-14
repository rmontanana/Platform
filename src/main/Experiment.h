#ifndef EXPERIMENT_H
#define EXPERIMENT_H
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <string>
#include <folding.hpp>
#include "bayesnet/BaseClassifier.h"
#include "HyperParameters.h"
#include "results/Result.h"
#include "bayesnet/network/Network.h"

namespace platform {
    using json = nlohmann::ordered_json;
    enum class score_t { NONE, ACCURACY, ROC_AUC_OVR };
    class Experiment {
    public:
        Experiment() = default;
        Experiment& setPlatform(const std::string& platform) { this->result.setPlatform(platform); return *this; }
        Experiment& setScoreName(const std::string& score_name) { this->result.setScoreName(score_name); return *this; }
        Experiment& setTitle(const std::string& title) { this->result.setTitle(title); return *this; }
        Experiment& setModelVersion(const std::string& model_version) { this->result.setModelVersion(model_version); return *this; }
        Experiment& setModel(const std::string& model) { this->result.setModel(model); return *this; }
        Experiment& setLanguage(const std::string& language) { this->result.setLanguage(language); return *this; }
        Experiment& setDiscretizationAlgorithm(const std::string& discretization_algo)
        {
            this->discretization_algo = discretization_algo; this->result.setDiscretizationAlgorithm(discretization_algo); return *this;
        }
        Experiment& setSmoothSrategy(const std::string& smooth_strategy);
        Experiment& setLanguageVersion(const std::string& language_version) { this->result.setLanguageVersion(language_version); return *this; }
        Experiment& setDiscretized(bool discretized) { this->discretized = discretized; result.setDiscretized(discretized); return *this; }
        Experiment& setStratified(bool stratified) { this->stratified = stratified; result.setStratified(stratified); return *this; }
        Experiment& setNFolds(int nfolds) { this->nfolds = nfolds; result.setNFolds(nfolds); return *this; }
        Experiment& addResult(PartialResult result_) { result.addPartial(result_); return *this; }
        Experiment& addRandomSeed(int randomSeed) { randomSeeds.push_back(randomSeed); result.addSeed(randomSeed); return *this; }
        Experiment& setDuration(float duration) { this->result.setDuration(duration); return *this; }
        Experiment& setHyperparameters(const HyperParameters& hyperparameters_) { this->hyperparameters = hyperparameters_; return *this; }
        HyperParameters& getHyperParameters() { return hyperparameters; }
        std::string getModel() const { return result.getModel(); }
        std::string getScore() const { return result.getScoreName(); }
        bool isDiscretized() const { return discretized; }
        bool isStratified() const { return stratified; }
        bool isQuiet() const { return quiet; }
        std::string getSmoothStrategy() const { return smooth_strategy; }
        int getNFolds() const { return nfolds; }
        std::vector<int> getRandomSeeds() const { return randomSeeds; }
        void cross_validation(const std::string& fileName);
        void go();
        void saveResult(const std::string& path);
        void show();
        void saveGraph();
        void report();
        void setFilesToTest(const std::vector<std::string>& filesToTest) { this->filesToTest = filesToTest; }
        void setQuiet(bool quiet) { this->quiet = quiet; }
        void setNoTrainScore(bool no_train_score) { this->no_train_score = no_train_score; }
        void setGenerateFoldFiles(bool generate_fold_files) { this->generate_fold_files = generate_fold_files; }
        void setGraph(bool graph) { this->graph = graph; }
    private:
        score_t parse_score() const;
        Result result;
        bool discretized{ false }, stratified{ false }, generate_fold_files{ false }, graph{ false }, quiet{ false }, no_train_score{ false };
        std::vector<PartialResult> results;
        std::vector<int> randomSeeds;
        std::vector<std::string> filesToTest;
        std::string discretization_algo;
        std::string smooth_strategy;
        bayesnet::Smoothing_t smooth_type{ bayesnet::Smoothing_t::NONE };
        HyperParameters hyperparameters;
        int nfolds{ 0 };
        int max_name{ 7 }; // max length of dataset name for formatting (default 7)
    };
}
#endif