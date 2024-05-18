#ifndef REPORTBASE_H
#define REPORTBASE_H
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include "common/Paths.h"
#include "common/Symbols.h"

namespace platform {
    using json = nlohmann::ordered_json;
    class ReportBase {
    public:
        explicit ReportBase(json data_, bool compare);
        virtual ~ReportBase() = default;
        void show();
    protected:
        json data;
        std::string fromVector(const std::string& key);
        std::string fVector(const std::string& title, const json& data, const int width, const int precision);
        bool getExistBestFile();
        virtual void header() = 0;
        virtual void body() = 0;
        virtual void showSummary() = 0;
        std::string compareResult(const std::string& dataset, double result);
        std::map<std::string, int> summary;
        double margin;
        std::map<std::string, std::string> meaning;
        bool compare;
        std::string nodes_label;
        std::string leaves_label;
        std::string depth_label;
    private:
        double bestResult(const std::string& dataset, const std::string& model);
        json bestResults;
        bool existBestFile = true;
    };
};
#endif