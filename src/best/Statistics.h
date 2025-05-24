#ifndef STATISTICS_H
#define STATISTICS_H
#include <iostream>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::ordered_json;

    struct WTL {
        uint win;
        uint tie;
        uint loss;
    };
    struct FriedmanResult {
        double statistic;
        double criticalValue;
        long double pvalue;
        bool reject;
    };
    struct PostHocLine {
        uint idx; //index of the main order
        std::string model;
        long double pvalue;
        double rank;
        WTL wtl;
        bool reject;
    };

    class Statistics {
    public:
        Statistics(const std::string& score, const std::vector<std::string>& models, const std::vector<std::string>& datasets, const json& data, double significance = 0.05, bool output = true);
        bool friedmanTest();
        void postHocTest();
        void postHocTestReport(bool friedmanResult, bool tex);
        int getControlIdx();
        FriedmanResult& getFriedmanResult() { return friedmanResult; }
        std::vector<PostHocLine>& getPostHocResults() { return postHocResults; }
        std::map<std::string, std::map<std::string, float>>& getRanks() { return ranksModels; } // ranks of the models per dataset
    private:
        void fit();
        void postHocHolmTest();
        void postHocWilcoxonTest();
        void computeRanks();
        void computeWTL();
        void Holm_Bonferroni();
        void setResultsOrder(); // Set the order of the results based on the statistic analysis needed
        void restoreResultsOrder(); // Restore the order of the results after the Holm-Bonferroni adjustment
        const std::string& score;
        std::string postHocType;
        const std::vector<std::string>& models;
        const std::vector<std::string>& datasets;
        const json& data;
        double significance;
        bool output;
        bool fitted = false;
        int nModels = 0;
        int nDatasets = 0;
        int controlIdx = 0;
        int greaterAverage = -1; // The model with the greater average score
        std::map<int, WTL> wtl;
        std::map<std::string, float> ranks;
        int maxModelName = 0;
        int maxDatasetName = 0;
        int hlen; // length of the line
        FriedmanResult friedmanResult;
        std::vector<PostHocLine> postHocResults;
        std::map<std::string, std::map<std::string, float>> ranksModels;
    };
}
#endif