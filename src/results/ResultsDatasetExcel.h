#pragma once

#include <nlohmann/json.hpp>
#include "reports/ExcelFile.h"

using json = nlohmann::json;

namespace platform {

    class ResultsDatasetExcel : public ExcelFile {
    public:
        ResultsDatasetExcel();
        ~ResultsDatasetExcel();
        void report(json& data);
    };
}
