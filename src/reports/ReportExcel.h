#ifndef REPORT_EXCEL_H
#define REPORT_EXCEL_H
#include <algorithm>
#include "main/Scores.h"
#include "common/Colors.h"
#include "ReportBase.h"
#include "ExcelFile.h"
namespace platform {
    using json = nlohmann::ordered_json;

    class ReportExcel : public ReportBase, public ExcelFile {
    public:
        explicit ReportExcel(json data_, bool compare, lxw_workbook* workbook, lxw_worksheet* worksheet = NULL);
        void closeFile();
    private:
        void formatColumns();
        void createFile();
        void header() override;
        void body() override;
        void showSummary() override;
        void footer(double totalScore, int row);
        void append_notes(const json& r, int row);
        void create_classification_report(const json& result);
        std::pair<int, int> write_classification_report(const json& result, int init_row, int init_col);
        void header_notes(int row);
    };
};
#endif