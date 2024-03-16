#pragma once

#include <locale>
#include <string>
#include <sstream>
#include <nlohmann/json.hpp>
#include "results/ResultsDataset.h"

namespace platform {
    class ResultsDatasetsConsole {
    public:
        ResultsDatasetsConsole() = default;
        ~ResultsDatasetsConsole() = default;
        std::string getOutput() const { return output.str(); }
        int getNumLines() const { return numLines; }
        json getData() { return data; }
        void list_results(const std::string& dataset, const std::string& score, const std::string& model);
    private:
        std::stringstream output;
        json data;
        int numLines = 0;
    };
}




