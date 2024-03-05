#pragma once
#include "ReportExcel.h"
namespace platform {
    class ReportExcelCompared {
    public:
        explicit ReportExcelCompared(json& data_A, json& data_B);
        ~ReportExcelCompared();
        void report();
    private:
        json& data_A;
        json& data_B;
        lxw_workbook* workbook;
    };
};