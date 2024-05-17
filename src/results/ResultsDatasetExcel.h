#pragma once

#include <nlohmann/json.hpp>
#include "reports/ExcelFile.h"


namespace platform {
    using json = nlohmann::ordered_json;

    class ResultsDatasetExcel : public ExcelFile {
    public:
        ResultsDatasetExcel();
        ~ResultsDatasetExcel();
        void report(json& data);
    };
}
