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
        DatasetsExcel();
        ~DatasetsExcel();
        void report();
    private:
        void formatColumns();
        int datasetNameSize = 25; // Min size of the column
    };
}
#endif //DATASETS_EXCEL_H