#ifndef STATISTICS_H
#define STATISTICS_H
#include <iostream>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::ordered_json;

    struct WTL {
        int win;
        int tie;
        int loss;
    };
    struct FriedmanResult {
        double statistic;
        double criticalValue;
        long double pvalue;
        bool reject;
    };
    struct PostHocLine {
        std::string model;
        long double pvalue;
        double rank;
        WTL wtl;
        bool reject;
    };
    struct PostHocResult {
        std::string model;
        std::vector<PostHocLine> postHocLines;
    };
    class Statistics {
    public:
        Statistics(const std::string& score, const std::vector<std::string>& models, const std::vector<std::string>& datasets, const json& data, double significance = 0.05, bool output = true);
        bool friedmanTest();
        void postHocTest();
        void postHocTestReport(bool friedmanResult, bool tex);
        int getControlIdx();
        FriedmanResult& getFriedmanResult();
        PostHocResult& getPostHocResult();
        std::map<std::string, std::map<std::string, float>>& getRanks();
    private:
        void fit();
        void postHocHolmTest();
        void postHocWilcoxonTest();
        void computeRanks();
        void computeWTL();
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
        std::vector<std::pair<int, double>> postHocData;
        int maxModelName = 0;
        int maxDatasetName = 0;
        int hlen; // length of the line
        FriedmanResult friedmanResult;
        PostHocResult postHocResult;
        std::map<std::string, std::map<std::string, float>> ranksModels;
    };
}
#endif