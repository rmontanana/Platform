#pragma once

#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "results/Result.h"
namespace platform {
    using json = nlohmann::ordered_json;
    enum class SortType {
        ASC = 0,
        DESC = 1,
    };
    enum class SortField {
        DATE = 0,
        MODEL = 1,
        SCORE = 2,
        DURATION = 3,
    };
    class ResultsManager {
    public:
        ResultsManager(const std::string& model, const std::string& score, const std::string& platform, bool complete, bool partial);
        void load(); // Loads the list of results
        void sortResults(SortField field, SortType type); // Sorts the list of results
        void sortDate(SortType type);
        void sortScore(SortType type);
        void sortModel(SortType type);
        void sortDuration(SortType type);
        int maxModelSize() const { return maxModel; };
        int maxTitleSize() const { return maxTitle; };
        void hideResult(int index, const std::string& pathHidden);
        void deleteResult(int index);
        int size() const;
        bool empty() const;
        std::vector<Result>::iterator begin() { return files.begin(); };
        std::vector<Result>::iterator end() { return files.end(); };
        Result& at(int index) { return files.at(index); };
    private:
        std::string path;
        std::string model;
        std::string scoreName;
        std::string platform;
        bool complete;
        bool partial;
        int maxModel;
        int maxTitle;
        std::vector<Result> files;
    };
};