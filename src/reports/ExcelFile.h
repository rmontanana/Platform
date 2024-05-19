#ifndef EXCELFILE_H
#define EXCELFILE_H
#include <locale>
#include <string>
#include <map>
#include <xlsxwriter.h>

namespace platform {
    class ExcelFile {
    public:
        ExcelFile();
        ExcelFile(lxw_workbook* workbook);
        ExcelFile(lxw_workbook* workbook, lxw_worksheet* worksheet);
        lxw_workbook* getWorkbook();
        std::string getFileName();
    protected:
        void setProperties(std::string title);
        void writeString(int row, int col, const std::string& text, const std::string& style = "");
        void writeInt(int row, int col, const int number, const std::string& style = "");
        void writeDouble(int row, int col, const double number, const std::string& style = "");
        void createFormats();
        void boldFontColor(const uint32_t color); // the same color for bold styles
        void boldRed(); //set red color for the bold styles
        void boldBlue(); //set blue color for the bold styles
        void boldGreen(); //set green color for the bold styles
        void createStyle(const std::string& name, lxw_format* style, bool odd);
        lxw_worksheet* createWorksheet(const std::string& name);
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
        std::string file_name;
    private:
        void setDefault();
    };
}
#endif