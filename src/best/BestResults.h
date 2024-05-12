#pragma once

#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;
namespace platform {
    class BestResults {
    public:
        explicit BestResults(const std::string& path, const std::string& score, const std::string& model, const std::string& dataset, bool friedman, double significance = 0.05)
            : path(path), score(score), model(model), dataset(dataset), friedman(friedman), significance(significance)
        {
        }
        std::string build();
        void reportSingle(bool excel);
        void reportAll(bool excel);
        void buildAll();
    private:
        std::vector<std::string> getModels();
        std::vector<std::string> getDatasets(json table);
        std::vector<std::string> loadResultFiles();
        void messageExcelFile(const std::string& fileName);
        json buildTableResults(std::vector<std::string> models);
        void printTableResults(std::vector<std::string> models, json table);
        std::string bestResultFile();
        json loadFile(const std::string& fileName);
        void listFile();
        std::string path;
        std::string score;
        std::string model;
        std::string dataset;
        bool friedman;
        double significance;
        int maxModelName = 0;
        int maxDatasetName = 0;
    };
}
