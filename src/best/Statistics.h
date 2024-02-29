#ifndef STATISTICS_H
#define STATISTICS_H
#include <iostream>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace platform {
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
    struct HolmLine {
        std::string model;
        long double pvalue;
        double rank;
        WTL wtl;
        bool reject;
    };
    struct HolmResult {
        std::string model;
        std::vector<HolmLine> holmLines;
    };
    class Statistics {
    public:
        Statistics(const std::vector<std::string>& models, const std::vector<std::string>& datasets, const json& data, double significance = 0.05, bool output = true);
        bool friedmanTest();
        void postHocHolmTest(bool friedmanResult);
        FriedmanResult& getFriedmanResult();
        HolmResult& getHolmResult();
        std::map<std::string, std::map<std::string, float>>& getRanks();
    private:
        void fit();
        void computeRanks();
        void computeWTL();
        const std::vector<std::string>& models;
        const std::vector<std::string>& datasets;
        const json& data;
        double significance;
        bool output;
        bool fitted = false;
        int nModels = 0;
        int nDatasets = 0;
        int controlIdx = 0;
        std::map<int, WTL> wtl;
        std::map<std::string, float> ranks;
        int maxModelName = 0;
        int maxDatasetName = 0;
        FriedmanResult friedmanResult;
        HolmResult holmResult;
        std::map<std::string, std::map<std::string, float>> ranksModels;
    };
}
#endif // !STATISTICS_H