#pragma once

#include <locale>
#include <sstream>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::json;


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

