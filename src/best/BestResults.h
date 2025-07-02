#ifndef BESTRESULTS_H
#define BESTRESULTS_H
#include <string>
#include <nlohmann/json.hpp>
namespace platform {
    using json = nlohmann::ordered_json;

    class BestResults {
    public:
        explicit BestResults(const std::string& path, const std::string& score, const std::string& model, const std::string& dataset, bool friedman, double significance = 0.05)
            : path(path), score(score), model(model), dataset(dataset), friedman(friedman), significance(significance)
        {
        }
        std::string build();
        void reportSingle(bool excel);
        void reportAll(bool excel, bool tex, bool index);
        void buildAll();
        std::string getExcelFileName() const { return excelFileName; }
    private:
        std::vector<std::string> getModels();
        std::vector<std::string> getDatasets(json table);
        std::vector<std::string> loadResultFiles();
        void messageOutputFile(const std::string& title, const std::string& fileName);
        json buildTableResults(std::vector<std::string> models);
        void printTableResults(std::vector<std::string> models, json table, bool tex, bool index);
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
        int minLength = 13; // Minimum length for scores
        std::string excelFileName;
    };
}
#endif