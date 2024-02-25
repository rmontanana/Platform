#ifndef RESULTS_H
#define RESULTS_H
#include <map>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "Result.h"
namespace platform {
    using json = nlohmann::json;
    class Results {
    public:
        Results(const std::string& path, const std::string& model, const std::string& score, bool complete, bool partial);
        void sortDate();
        void sortScore();
        void sortModel();
        void sortDuration();
        int maxModelSize() const { return maxModel; };
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
        bool complete;
        bool partial;
        int maxModel;
        std::vector<Result> files;
        void load(); // Loads the list of results
    };
};
#endif