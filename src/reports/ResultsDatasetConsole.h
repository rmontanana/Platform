#pragma once

#include <locale>
#include <string>
#include <sstream>
#include <nlohmann/json.hpp>
#include "results/ResultsDataset.h"
#include "ReportsPaged.h"

namespace platform {
    class ResultsDatasetsConsole : public ReportsPaged {
    public:
        ResultsDatasetsConsole() = default;
        ~ResultsDatasetsConsole() = default;
        void report(const std::string& dataset, const std::string& score, const std::string& model);
    };
}




