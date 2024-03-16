#pragma once

#include <locale>
#include <sstream>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::json;

    struct separated_datasets : numpunct<char> {
        char do_decimal_point() const { return ','; }
        char do_thousands_sep() const { return '.'; }
        std::string do_grouping() const { return "\03"; }
    };

    class DatasetsConsole {
    public:
        static const int BALANCE_LENGTH;
        DatasetsConsole() = default;
        ~DatasetsConsole() = default;
        std::string getOutput() const { return output.str(); }
        int getNumLines() const { return numLines; }
        json& getData() { return data; }
        std::string outputBalance(const std::string& balance);
        void list_datasets();
    private:
        std::stringstream output;
        json data;
        int numLines = 0;
    };
}

