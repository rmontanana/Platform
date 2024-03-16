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
        std::string getOutput() const;
        std::string getHeader() const;
        std::vector<std::string>& getBody() { return body; }
        int getNumLines() const { return body.size(); }
        json& getData() { return data; }
        void report();
    private:
        void split_lines(int name_len, std::string line, const std::string& balance);
        std::vector<std::string> header, body;
        json data;
    };
}

