#ifndef DATASETS_EXCEL_H
#define DATASETS_EXCEL_H
#include "ExcelFile.h"
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace platform {

    class DatasetsExcel : public ExcelFile {
    public:
        explicit DatasetsExcel(json& data);
        ~DatasetsExcel();
        void report();
    private:
        void formatColumns(int dataset, int balance);
        json data;

    };
}
#endif //DATASETS_EXCEL_H