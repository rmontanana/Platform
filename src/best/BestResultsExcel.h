#ifndef BESTRESULTSEXCEL_H
#define BESTRESULTSEXCEL_H
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include "reports/ExcelFile.h"


namespace platform {
    using json = nlohmann::ordered_json;
    class BestResultsExcel : public ExcelFile {
    public:
        BestResultsExcel(const std::string& score, const std::vector<std::string>& datasets);
        ~BestResultsExcel();
        void reportAll(const std::vector<std::string>& models, const json& table, const std::map<std::string, std::map<std::string, float>>& ranks, bool friedman, double significance);
        void reportSingle(const std::string& model, const std::string& fileName);
    private:
        void build();
        void header(bool ranks);
        void body(bool ranks);
        void footer(bool ranks);
        void formatColumns();
        void doFriedman();
        void addConditionalFormat(std::string formula);
        std::string score;
        std::vector<std::string> models;
        std::vector<std::string> datasets;
        json table;
        std::map<std::string, std::map<std::string, float>> ranksModels;
        bool friedman;
        double significance;
        int modelNameSize = 12; // Min size of the column
        int datasetNameSize = 25; // Min size of the column
    };
}
#endif