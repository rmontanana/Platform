#include "ExcelFile.h"

namespace platform {
    ExcelFile::ExcelFile()
    {
        setDefault();
    }
    ExcelFile::ExcelFile(lxw_workbook* workbook) : workbook(workbook)
    {
        setDefault();
    }
    ExcelFile::ExcelFile(lxw_workbook* workbook, lxw_worksheet* worksheet) : workbook(workbook), worksheet(worksheet)
    {
        setDefault();
    }
    void ExcelFile::setDefault()
    {
        normalSize = 14; //font size for report body
        row = 0;
        colorTitle = 0xB1A0C7;
        colorOdd = 0xDCE6F1;
        colorEven = 0xFDE9D9;
    }

    lxw_workbook* ExcelFile::getWorkbook()
    {
        return workbook;
    }
    void ExcelFile::setProperties(std::string title)
    {
        char line[title.size() + 1];
        strcpy(line, title.c_str());
        lxw_doc_properties properties = {
            .title = line,
            .subject = (char*)"Machine learning results",
            .author = (char*)"Ricardo Montañana Gómez",
            .manager = (char*)"Dr. J. A. Gámez, Dr. J. M. Puerta",
            .company = (char*)"UCLM",
            .comments = (char*)"Created with libxlsxwriter and c++",
        };
        workbook_set_properties(workbook, &properties);
    }
    lxw_format* ExcelFile::efectiveStyle(const std::string& style)
    {
        lxw_format* efectiveStyle = NULL;
        if (style != "") {
            std::string suffix = row % 2 ? "_odd" : "_even";
            try {
                efectiveStyle = styles.at(style + suffix);
            }
            catch (const std::out_of_range& oor) {
                try {
                    efectiveStyle = styles.at(style);
                }
                catch (const std::out_of_range& oor) {
                    throw std::invalid_argument("Style " + style + " not found");
                }
            }
        }
        return efectiveStyle;
    }
    void ExcelFile::writeString(int row, int col, const std::string& text, const std::string& style)
    {
        worksheet_write_string(worksheet, row, col, text.c_str(), efectiveStyle(style));
    }
    void ExcelFile::writeInt(int row, int col, const int number, const std::string& style)
    {
        worksheet_write_number(worksheet, row, col, number, efectiveStyle(style));
    }
    void ExcelFile::writeDouble(int row, int col, const double number, const std::string& style)
    {
        worksheet_write_number(worksheet, row, col, number, efectiveStyle(style));
    }
    void ExcelFile::addColor(lxw_format* style, bool odd)
    {
        uint32_t efectiveColor = odd ? colorEven : colorOdd;
        format_set_bg_color(style, lxw_color_t(efectiveColor));
    }
    void ExcelFile::createStyle(const std::string& name, lxw_format* style, bool odd)
    {
        addColor(style, odd);
        if (name == "textCentered") {
            format_set_align(style, LXW_ALIGN_CENTER);
            format_set_font_size(style, normalSize);
            format_set_border(style, LXW_BORDER_THIN);
        } else if (name == "text") {
            format_set_font_size(style, normalSize);
            format_set_border(style, LXW_BORDER_THIN);
        } else if (name == "bodyHeader") {
            format_set_bold(style);
            format_set_font_size(style, normalSize);
            format_set_align(style, LXW_ALIGN_CENTER);
            format_set_align(style, LXW_ALIGN_VERTICAL_CENTER);
            format_set_border(style, LXW_BORDER_THIN);
            format_set_bg_color(style, lxw_color_t(colorTitle));
        } else if (name == "result") {
            format_set_font_size(style, normalSize);
            format_set_border(style, LXW_BORDER_THIN);
            format_set_num_format(style, "0.0000000");
        } else if (name == "time") {
            format_set_font_size(style, normalSize);
            format_set_border(style, LXW_BORDER_THIN);
            format_set_num_format(style, "#,##0.000000");
        } else if (name == "ints") {
            format_set_font_size(style, normalSize);
            format_set_num_format(style, "###,##0");
            format_set_border(style, LXW_BORDER_THIN);
        } else if (name == "floats") {
            format_set_border(style, LXW_BORDER_THIN);
            format_set_font_size(style, normalSize);
            format_set_num_format(style, "#,##0.00");
        }
    }

    void ExcelFile::createFormats()
    {
        auto styleNames = { "text", "textCentered", "bodyHeader", "result", "time", "ints", "floats" };
        lxw_format* style;
        for (std::string name : styleNames) {
            lxw_format* style = workbook_add_format(workbook);
            style = workbook_add_format(workbook);
            createStyle(name, style, true);
            styles[name + "_odd"] = style;
            style = workbook_add_format(workbook);
            createStyle(name, style, false);
            styles[name + "_even"] = style;
        }

        // Header 1st line
        lxw_format* headerFirst = workbook_add_format(workbook);
        format_set_bold(headerFirst);
        format_set_font_size(headerFirst, 18);
        format_set_align(headerFirst, LXW_ALIGN_CENTER);
        format_set_align(headerFirst, LXW_ALIGN_VERTICAL_CENTER);
        format_set_border(headerFirst, LXW_BORDER_THIN);
        format_set_bg_color(headerFirst, lxw_color_t(colorTitle));

        // Header rest
        lxw_format* headerRest = workbook_add_format(workbook);
        format_set_bold(headerRest);
        format_set_align(headerRest, LXW_ALIGN_CENTER);
        format_set_font_size(headerRest, 16);
        format_set_align(headerRest, LXW_ALIGN_VERTICAL_CENTER);
        format_set_border(headerRest, LXW_BORDER_THIN);
        format_set_bg_color(headerRest, lxw_color_t(colorOdd));

        // Header small
        lxw_format* headerSmall = workbook_add_format(workbook);
        format_set_bold(headerSmall);
        format_set_align(headerSmall, LXW_ALIGN_LEFT);
        format_set_font_size(headerSmall, 12);
        format_set_border(headerSmall, LXW_BORDER_THIN);
        format_set_align(headerSmall, LXW_ALIGN_VERTICAL_CENTER);
        format_set_bg_color(headerSmall, lxw_color_t(colorOdd));

        // Summary style
        lxw_format* summaryStyle = workbook_add_format(workbook);
        format_set_bold(summaryStyle);
        format_set_font_size(summaryStyle, 16);
        format_set_border(summaryStyle, LXW_BORDER_THIN);
        format_set_align(summaryStyle, LXW_ALIGN_VERTICAL_CENTER);

        styles["headerFirst"] = headerFirst;
        styles["headerRest"] = headerRest;
        styles["headerSmall"] = headerSmall;
        styles["summaryStyle"] = summaryStyle;
    }
}