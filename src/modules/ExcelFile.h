#ifndef EXCELFILE_H
#define EXCELFILE_H
#include <locale>
#include <string>
#include <map>
#include "xlsxwriter.h"

namespace platform {
    struct separated : std::numpunct<char> {
        char do_decimal_point() const { return ','; }

        char do_thousands_sep() const { return '.'; }

        std::string do_grouping() const { return "\03"; }
    };
    class ExcelFile {
    public:
        ExcelFile();
        ExcelFile(lxw_workbook* workbook);
        ExcelFile(lxw_workbook* workbook, lxw_worksheet* worksheet);
        lxw_workbook* getWorkbook();
    protected:
        void setProperties(std::string title);
        void writeString(int row, int col, const std::string& text, const std::string& style = "");
        void writeInt(int row, int col, const int number, const std::string& style = "");
        void writeDouble(int row, int col, const double number, const std::string& style = "");
        void createFormats();
        void createStyle(const std::string& name, lxw_format* style, bool odd);
        void addColor(lxw_format* style, bool odd);
        lxw_format* efectiveStyle(const std::string& name);
        lxw_workbook* workbook;
        lxw_worksheet* worksheet;
        std::map<std::string, lxw_format*> styles;
        int row;
        int normalSize; //font size for report body
        uint32_t colorTitle;
        uint32_t colorOdd;
        uint32_t colorEven;
    private:
        void setDefault();
    };
}
#endif // !EXCELFILE_H