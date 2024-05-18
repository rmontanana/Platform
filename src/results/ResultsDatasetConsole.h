#ifndef RESULTSDATASETSCONSOLE_H
#define RESULTSDATASETSCONSOLE_H
#include <locale>
#include <string>
#include <sstream>
#include <nlohmann/json.hpp>
#include "reports/ReportsPaged.h"
#include "ResultsDataset.h"

namespace platform {
    class ResultsDatasetsConsole : public ReportsPaged {
    public:
        ResultsDatasetsConsole() = default;
        ~ResultsDatasetsConsole() = default;
        bool report(const std::string& dataset, const std::string& score, const std::string& model);
    };
}
#endif