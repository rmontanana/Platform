#pragma once

#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class PartialResult {

public:
    PartialResult() { data["scores_train"] = json::array(); data["scores_test"] = json::array(); data["times_train"] = json::array(); data["times_test"] = json::array(); };
    PartialResult& setDataset(const std::string& dataset) { data["dataset"] = dataset; return *this; }
    PartialResult& setNotes(const std::vector<std::string>& notes) { this->notes.insert(this->notes.end(), notes.begin(), notes.end()); return *this; }
    PartialResult& setHyperparameters(const json& hyperparameters) { data["hyperparameters"] = hyperparameters; return *this; }
    PartialResult& setSamples(int samples) { data["samples"] = samples; return *this; }
    PartialResult& setFeatures(int features) { data["features"] = features; return *this; }
    PartialResult& setClasses(int classes) { data["classes"] = classes; return *this; }
    PartialResult& setScoreTrain(double score) { data["score_train"] = score; return *this; }
    PartialResult& setScoreTrainStd(double score_std) { data["score_train_std"] = score_std; return *this; }
    PartialResult& setScoreTest(double score) { data["score"] = score; return *this; }
    PartialResult& setScoreTestStd(double score_std) { data["score_std"] = score_std; return *this; }
    PartialResult& setTrainTime(double train_time) { data["train_time"] = train_time; return *this; }
    PartialResult& setTrainTimeStd(double train_time_std) { data["train_time_std"] = train_time_std; return *this; }
    PartialResult& setTestTime(double test_time) { data["test_time"] = test_time; return *this; }
    PartialResult& setTestTimeStd(double test_time_std) { data["test_time_std"] = test_time_std; return *this; }
    PartialResult& setNodes(float nodes) { data["nodes"] = nodes; return *this; }
    PartialResult& setLeaves(float leaves) { data["leaves"] = leaves; return *this; }
    PartialResult& setDepth(float depth) { data["depth"] = depth; return *this; }
    PartialResult& addScoreTrain(double score) { data["scores_train"].push_back(score); return *this; }
    PartialResult& addScoreTest(double score) { data["scores_test"].push_back(score); return *this; }
    PartialResult& addTimeTrain(double time) { data["times_train"].push_back(time); return *this; }
    PartialResult& addTimeTest(double time) { data["times_test"].push_back(time); return *this; }
    json getJson()
    {
        data["time"] = data["test_time"].get<double>() + data["train_time"].get<double>();
        data["time_std"] = data["test_time_std"].get<double>() + data["train_time_std"].get<double>();
        data["notes"] = notes;
        return data;
    }
private:
    json data;
    std::vector<std::string> notes;
};