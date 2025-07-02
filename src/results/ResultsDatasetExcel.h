#ifndef RESULTSDATASETEXCEL_H
#define RESULTSDATASETEXCEL_H
#include <nlohmann/json.hpp>
#include "reports/ExcelFile.h"


namespace platform {
    using json = nlohmann::ordered_json;

    class ResultsDatasetExcel : public ExcelFile {
    public:
        ResultsDatasetExcel();
        ~ResultsDatasetExcel();
        void report(json& data);
        std::string getExcelFileName() { return getFileName(); }
    };
}
#endif