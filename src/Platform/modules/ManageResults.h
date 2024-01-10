#ifndef MANAGE_RESULTS_H
#define MANAGE_RESULTS_H
#include "Results.h"
#include "xlsxwriter.h"

namespace platform {
    class ManageResults {
    public:
        ManageResults(int numFiles, const std::string& model, const std::string& score, bool complete, bool partial, bool compare);
        ~ManageResults() = default;
        void doMenu();
    private:
        void list();
        bool confirmAction(const std::string& intent, const std::string& fileName) const;
        void report(const int index, const bool excelReport);
        void showIndex(const int index, const int idx);
        void sortList();
        void menu();
        int numFiles;
        bool indexList;
        bool openExcel;
        bool complete;
        bool partial;
        bool compare;
        Results results;
        lxw_workbook* workbook;
    };

}

#endif /* MANAGE_RESULTS_H */