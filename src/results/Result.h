#ifndef RESULT_H
#define RESULT_H
#include <map>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "common/Timer.hpp"
#include "main/HyperParameters.h"
#include "main/PartialResult.h"

namespace platform {
    using json = nlohmann::ordered_json;

    class Result {
    public:
        Result();
        Result& load(const std::string& path, const std::string& filename);
        void save(const std::string& path);
        std::vector<std::string> check();
        // Getters
        json getJson();
        std::string to_string(int maxModel, int maxTitle) const;
        std::string getFilename() const;
        std::string getDate() const { return data["date"].get<std::string>(); };
        std::string getTime() const { return data["time"].get<std::string>(); };
        double getScore() const { return score; };
        std::string getTitle() const { return data["title"].get<std::string>(); };
        double getDuration() const { return data["duration"]; };
        std::string getModel() const { return data["model"].get<std::string>(); };
        std::string getPlatform() const { return data["platform"].get<std::string>(); };
        std::string getScoreName() const { return data["score_name"].get<std::string>(); };
        void setSchemaVersion(const std::string& version) { data["schema_version"] = version; };
        bool isComplete() const { return complete; };
        json getData() const { return data; }
        // Setters
        void setTitle(const std::string& title) { data["title"] = title; };
        void setSmoothStrategy(const std::string& smooth_strategy) { data["smooth_strategy"] = smooth_strategy; };
        void setDiscretizationAlgorithm(const std::string& discretization_algo) { data["discretization_algorithm"] = discretization_algo; };
        void setLanguage(const std::string& language) { data["language"] = language; };
        void setLanguageVersion(const std::string& language_version) { data["language_version"] = language_version; };
        void setDuration(double duration) { data["duration"] = duration; };
        void setModel(const std::string& model) { data["model"] = model; };
        void setModelVersion(const std::string& model_version) { data["version"] = model_version; };
        void setScoreName(const std::string& scoreName) { data["score_name"] = scoreName; };
        void setDiscretized(bool discretized) { data["discretized"] = discretized; };
        void addSeed(int seed) { data["seeds"].push_back(seed); };
        void addPartial(PartialResult& partial_result) { data["results"].push_back(partial_result.getJson()); };
        void setStratified(bool stratified) { data["stratified"] = stratified; };
        void setNFolds(int nfolds) { data["folds"] = nfolds; };
        void setPlatform(const std::string& platform_name) { data["platform"] = platform_name; };
    private:
        json data;
        bool complete;
        double score = 0.0;
    };
};
#endif