#ifndef REPORTEXCEL_H
#define REPORTEXCEL_H
#include<map>
#include "xlsxwriter.h"
#include "ReportBase.h"
#include "ExcelFile.h"
#include "Colors.h"
namespace platform {
    class ReportExcel : public ReportBase, public ExcelFile {
    public:
        explicit ReportExcel(json data_, bool compare, lxw_workbook* workbook, lxw_worksheet* worksheet = NULL);
    private:
        void formatColumns();
        void createFile();
        void createWorksheet();
        void closeFile();
        void header() override;
        void body() override;
        void showSummary() override;
        void footer(double totalScore, int row);

    };
};
#endif // !REPORTEXCEL_H