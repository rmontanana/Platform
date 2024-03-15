#pragma once

#include <xlsxwriter.h>
#include "ResultsManager.h"
#include "Paginator.hpp"

namespace platform {
    class ManageResults {
    public:
        ManageResults(int numFiles, const std::string& model, const std::string& score, bool complete, bool partial, bool compare);
        ~ManageResults() = default;
        void doMenu();
    private:
        void list(const std::string& status, const std::string& color, int index_A, int index_B);
        bool confirmAction(const std::string& intent, const std::string& fileName) const;
        std::string report(const int index, const bool excelReport);
        std::string report_compared(const int index_A, const int index_B);
        void showIndex(const int index, const int idx);
        std::pair<std::string, std::string> sortList();
        void menu();
        int numFiles;
        bool indexList;
        bool openExcel;
        bool complete;
        bool partial;
        bool compare;
        std::string sort_field;
        int page;
        Paginator paginator;
        ResultsManager results;
        lxw_workbook* workbook;
    };
}
