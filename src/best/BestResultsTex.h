#ifndef BEST_RESULTS_TEX_H
#define BEST_RESULTS_TEX_H
#include <map>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/Paths.h"
#include "Statistics.h"
namespace platform {
    using json = nlohmann::ordered_json;
    class BestResultsTex {
    public:
        BestResultsTex(bool dataset_name = true) : dataset_name(dataset_name) {};
        ~BestResultsTex() = default;
        void results_header(const std::vector<std::string>& models, const std::string& date, bool index);
        void results_body(const std::vector<std::string>& datasets, json& table, bool index);
        void results_footer(const std::map<std::string, std::vector<double>>& totals, const std::string& best_model);
        void holm_test(struct HolmResult& holmResult, const std::string& date);
    private:
        bool dataset_name;
        void openTexFile(const std::string& name);
        std::ofstream handler;
        std::vector<std::string> models;
    };
}
#endif