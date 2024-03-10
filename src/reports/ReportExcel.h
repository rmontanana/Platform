#pragma once

#include <map>
#include <xlsxwriter.h>
#include "common/Colors.h"
#include "ReportBase.h"
#include "ExcelFile.h"
namespace platform {
    class ReportExcel : public ReportBase, public ExcelFile {
    public:
        explicit ReportExcel(json data_, bool compare, lxw_workbook* workbook, lxw_worksheet* worksheet = NULL);
        void closeFile();
    private:
        void formatColumns();
        void createFile();
        void createWorksheet();
        void header() override;
        void body() override;
        void showSummary() override;
        void footer(double totalScore, int row);
        void append_notes(const json& r, int row);
        void header_notes(int row);
    };
};
