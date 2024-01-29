#ifndef EXPERIMENT_H
#define EXPERIMENT_H
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <string>
#include "folding.hpp"
#include "BaseClassifier.h"
#include "HyperParameters.h"
#include "TAN.h"
#include "KDB.h"
#include "AODE.h"
#include "Timer.h"

namespace platform {
    using json = nlohmann::json;
    class Result {
    private:
        std::string dataset, model_version;
        json hyperparameters;
        int samples{ 0 }, features{ 0 }, classes{ 0 };
        double score_train{ 0 }, score_test{ 0 }, score_train_std{ 0 }, score_test_std{ 0 }, train_time{ 0 }, train_time_std{ 0 }, test_time{ 0 }, test_time_std{ 0 };
        float nodes{ 0 }, leaves{ 0 }, depth{ 0 };
        std::vector<double> scores_train, scores_test, times_train, times_test;
    public:
        Result() = default;
        Result& setDataset(const std::string& dataset) { this->dataset = dataset; return *this; }
        Result& setHyperparameters(const json& hyperparameters) { this->hyperparameters = hyperparameters; return *this; }
        Result& setSamples(int samples) { this->samples = samples; return *this; }
        Result& setFeatures(int features) { this->features = features; return *this; }
        Result& setClasses(int classes) { this->classes = classes; return *this; }
        Result& setScoreTrain(double score) { this->score_train = score; return *this; }
        Result& setScoreTest(double score) { this->score_test = score; return *this; }
        Result& setScoreTrainStd(double score_std) { this->score_train_std = score_std; return *this; }
        Result& setScoreTestStd(double score_std) { this->score_test_std = score_std; return *this; }
        Result& setTrainTime(double train_time) { this->train_time = train_time; return *this; }
        Result& setTrainTimeStd(double train_time_std) { this->train_time_std = train_time_std; return *this; }
        Result& setTestTime(double test_time) { this->test_time = test_time; return *this; }
        Result& setTestTimeStd(double test_time_std) { this->test_time_std = test_time_std; return *this; }
        Result& setNodes(float nodes) { this->nodes = nodes; return *this; }
        Result& setLeaves(float leaves) { this->leaves = leaves; return *this; }
        Result& setDepth(float depth) { this->depth = depth; return *this; }
        Result& addScoreTrain(double score) { scores_train.push_back(score); return *this; }
        Result& addScoreTest(double score) { scores_test.push_back(score); return *this; }
        Result& addTimeTrain(double time) { times_train.push_back(time); return *this; }
        Result& addTimeTest(double time) { times_test.push_back(time); return *this; }
        const float get_score_train() const { return score_train; }
        float get_score_test() { return score_test; }
        const std::string& getDataset() const { return dataset; }
        const json& getHyperparameters() const { return hyperparameters; }
        const int getSamples() const { return samples; }
        const int getFeatures() const { return features; }
        const int getClasses() const { return classes; }
        const double getScoreTrain() const { return score_train; }
        const double getScoreTest() const { return score_test; }
        const double getScoreTrainStd() const { return score_train_std; }
        const double getScoreTestStd() const { return score_test_std; }
        const double getTrainTime() const { return train_time; }
        const double getTrainTimeStd() const { return train_time_std; }
        const double getTestTime() const { return test_time; }
        const double getTestTimeStd() const { return test_time_std; }
        const float getNodes() const { return nodes; }
        const float getLeaves() const { return leaves; }
        const float getDepth() const { return depth; }
        const std::vector<double>& getScoresTrain() const { return scores_train; }
        const std::vector<double>& getScoresTest() const { return scores_test; }
        const std::vector<double>& getTimesTrain() const { return times_train; }
        const std::vector<double>& getTimesTest() const { return times_test; }
    };
    class Experiment {
    public:
        Experiment() = default;
        Experiment& setTitle(const std::string& title) { this->title = title; return *this; }
        Experiment& setModel(const std::string& model) { this->model = model; return *this; }
        Experiment& setPlatform(const std::string& platform) { this->platform = platform; return *this; }
        Experiment& setScoreName(const std::string& score_name) { this->score_name = score_name; return *this; }
        Experiment& setModelVersion(const std::string& model_version) { this->model_version = model_version; return *this; }
        Experiment& setLanguage(const std::string& language) { this->language = language; return *this; }
        Experiment& setLanguageVersion(const std::string& language_version) { this->language_version = language_version; return *this; }
        Experiment& setDiscretized(bool discretized) { this->discretized = discretized; return *this; }
        Experiment& setStratified(bool stratified) { this->stratified = stratified; return *this; }
        Experiment& setNFolds(int nfolds) { this->nfolds = nfolds; return *this; }
        Experiment& addResult(Result result) { results.push_back(result); return *this; }
        Experiment& addRandomSeed(int randomSeed) { randomSeeds.push_back(randomSeed); return *this; }
        Experiment& setDuration(float duration) { this->duration = duration; return *this; }
        Experiment& setHyperparameters(const HyperParameters& hyperparameters_) { this->hyperparameters = hyperparameters_; return *this; }
        std::string get_file_name();
        void save(const std::string& path);
        void cross_validation(const std::string& fileName, bool quiet);
        void go(std::vector<std::string> filesToProcess, bool quiet);
        void show();
        void report();
    private:
        std::string title, model, platform, score_name, model_version, language_version, language;
        bool discretized{ false }, stratified{ false };
        std::vector<Result> results;
        std::vector<int> randomSeeds;
        HyperParameters hyperparameters;
        int nfolds{ 0 };
        int max_name{ 7 }; // max length of dataset name for formatting (default 7)
        float duration{ 0 };
        json build_json();
    };
}
#endif