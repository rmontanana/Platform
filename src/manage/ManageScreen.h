#ifndef MANAGE_SCREEN_H
#define MANAGE_SCREEN_H
#include <xlsxwriter.h>
#include "ResultsManager.h"
#include "common/Colors.h"
#include "Paginator.hpp"

namespace platform {
    enum class OutputType {
        EXPERIMENTS = 0,
        DATASETS = 1,
        RESULT = 2,
        DETAIL = 3,
        Count
    };
    class ManageScreen {
    public:
        ManageScreen(int rows, int cols, const std::string& model, const std::string& score, const std::string& platform, bool complete, bool partial, bool compare);
        ~ManageScreen() = default;
        void doMenu();
        void updateSize(int rows, int cols);
    private:
        void list(const std::string& status, const std::string& color);
        void list_experiments(const std::string& status, const std::string& color);
        void list_result(const std::string& status, const std::string& color);
        void list_detail(const std::string& status, const std::string& color);
        void list_datasets(const std::string& status, const std::string& color);
        bool confirmAction(const std::string& intent, const std::string& fileName) const;
        std::string report(const int index, const bool excelReport);
        std::string report_compared();
        std::pair<std::string, std::string> sortList();
        std::string getVersions();
        void computeSizes();
        void menu();
        void header();
        void footer(const std::string& status, const std::string& color);
        OutputType output_type;
        int rows;
        int cols;
        int min_columns;
        int index;
        int subIndex;
        int index_A, index_B; // used for comparison of experiments
        bool indexList;
        bool openExcel;
        bool didExcel;
        bool complete;
        bool partial;
        bool compare;
        bool resize = false;
        int maxModel, maxTitle;
        std::vector<std::string> header_labels;
        std::vector<int> header_lengths;
        std::vector<std::string> sort_fields;
        SortField sort_field = SortField::DATE;
        SortType sort_type = SortType::DESC;
        std::vector<Paginator> paginator;
        ResultsManager results;
        lxw_workbook* workbook;
    };
}
#endif