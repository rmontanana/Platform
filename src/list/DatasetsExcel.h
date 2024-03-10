#pragma once

#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include "reports/ExcelFile.h"

using json = nlohmann::json;

namespace platform {

    class DatasetsExcel : public ExcelFile {
    public:
        DatasetsExcel();
        ~DatasetsExcel();
        void report(json& data);
    };
}
