#ifndef BEST_RESULTS_TEX_H
#define BEST_RESULTS_TEX_H
#include <map>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/Paths.h"
namespace platform {
    using json = nlohmann::ordered_json;
    class BestResultsTex {
    public:
        BestResultsTex(const std::vector<std::string>& models, const std::string& date);
        ~BestResultsTex() = default;
        void results_header();
        void results_body(const std::vector<std::string>& datasets, json& table);
        void results_footer(const std::map<std::string, std::vector<double>>& totals, const std::string& best_model);
    private:
        std::FILE* output_tex;
        std::vector<std::string> models;
        std::string date;
    };
}
#endif