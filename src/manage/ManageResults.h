#pragma once

#include <xlsxwriter.h>
#include "ResultsManager.h"
#include "Paginator.hpp"

namespace platform {
    enum class OutputType {
        EXPERIMENTS = 0,
        DATASETS = 1,
        RESULT = 2,
        Count
    };
    class ManageResults {
    public:
        ManageResults(int numFiles, const std::string& model, const std::string& score, bool complete, bool partial, bool compare);
        ~ManageResults() = default;
        void doMenu();
    private:
        void list(const std::string& status, const std::string& color);
        void list_experiments(const std::string& status, const std::string& color);
        void list_result(const std::string& status, const std::string& color);
        void list_datasets(const std::string& status, const std::string& color);
        bool confirmAction(const std::string& intent, const std::string& fileName) const;
        std::string report(const int index, const bool excelReport);
        std::string report_compared();
        void showIndex(const int index, const int idx);
        std::pair<std::string, std::string> sortList();
        void menu();
        void header();
        void footer(const std::string& status, const std::string& color);
        OutputType output_type;
        int numFiles;
        int index_A, index_B; // used for comparison of experiments
        int max_status_line;
        bool indexList;
        bool openExcel;
        bool didExcel;
        bool complete;
        bool partial;
        bool compare;
        std::string sort_field;
        std::vector<Paginator> paginator;
        ResultsManager results;
        lxw_workbook* workbook;
    };
}
