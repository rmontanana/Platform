#ifndef DATASETSEXCEL_H
#define DATASETSEXCEL_H
#include <nlohmann/json.hpp>
#include "reports/ExcelFile.h"


namespace platform {
    using json = nlohmann::ordered_json;
    class DatasetsExcel : public ExcelFile {
    public:
        DatasetsExcel();
        ~DatasetsExcel();
        void report(json& data);
        std::string getExcelFileName() { return getFileName(); }
    };
}
#endif