#pragma once

#include <locale>
#include <sstream>
#include <nlohmann/json.hpp>
#include "ReportsPaged.h"

namespace platform {
    using json = nlohmann::json;


    class DatasetsConsole : public ReportsPaged {
    public:
        static const int BALANCE_LENGTH;
        DatasetsConsole() = default;
        ~DatasetsConsole() = default;
        void report();
    private:
        void split_lines(int name_len, std::string line, const std::string& balance);
    };
}

