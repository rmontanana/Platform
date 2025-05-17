#ifndef BEST_RESULTS_MD_H
#define BEST_RESULTS_MD_H
#include <map>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/Paths.h"
#include "Statistics.h"
namespace platform {
    using json = nlohmann::ordered_json;
    class BestResultsMd {
    public:
        BestResultsMd() = default;
        ~BestResultsMd() = default;
        void results_header(const std::vector<std::string>& models, const std::string& date);
        void results_body(const std::vector<std::string>& datasets, json& table);
        void results_footer(const std::map<std::string, std::vector<double>>& totals, const std::string& best_model);
        void postHoc_test(struct PostHocResult& postHocResult, const std::string& kind, const std::string& date);
    private:
        void openMdFile(const std::string& name);
        std::ofstream handler;
        std::vector<std::string> models;
    };
}
#endif