#pragma once
#include "ReportExcel.h"
namespace platform {
    class ReportExcelCompared : public ExcelFile {
    public:
        explicit ReportExcelCompared(json& data_A, json& data_B);
        ~ReportExcelCompared();
        void report();
    private:
        void header();
        void body();
        void footer(std::vector<double>& totals_A, std::vector<double>& totals_B, int row);
        json& data_A;
        json& data_B;
        std::string nodes_label;
        std::string leaves_label;
        std::string depth_label;
    };
};