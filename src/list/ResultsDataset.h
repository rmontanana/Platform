#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "main/Result.h"
namespace platform {
    using json = nlohmann::json;
    class ResultsDataset {
    public:
        ResultsDataset(const std::string& dataset, const std::string& model, const std::string& score);
        void load(); // Loads the list of results
        void sortModel();
        int maxModelSize() const { return maxModel; };
        int maxFileSize() const { return maxFile; };
        int maxHyperSize() const { return maxHyper; };
        double maxResultScore() const { return maxResult; };
        int size() const;
        bool empty() const;
        std::vector<Result>::iterator begin() { return files.begin(); };
        std::vector<Result>::iterator end() { return files.end(); };
        Result& at(int index) { return files.at(index); };
    private:
        std::string path;
        std::string dataset;
        std::string model;
        std::string scoreName;
        int maxModel;
        int maxFile;
        int maxHyper;
        double maxResult;
        std::vector<Result> files;
    };
};