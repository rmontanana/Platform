#include "ReportExcelCompared.h"

namespace platform {

    ReportExcelCompared::ReportExcelCompared(json& data_A, json& data_B) : data_A(data_A), data_B(data_B), workbook(NULL)
    {
        ReportExcel report(data_A, false, workbook);
        workbook = report.getWorkbook();
        report.show();
        report = ReportExcel(data_B, false, workbook);
        report.show();
    }
    ReportExcelCompared::~ReportExcelCompared()
    {
        workbook_close(workbook);
    }
    void ReportExcelCompared::report()
    {

    }
}