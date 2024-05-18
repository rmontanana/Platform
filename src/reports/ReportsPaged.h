#ifndef REPORTS_PAGED_H
#define REPORTS_PAGED_H
#include <locale>
#include <sstream>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::ordered_json;

    class ReportsPaged {
    public:
        ReportsPaged();
        ~ReportsPaged() = default;
        std::string getOutput() const;
        std::string getHeader() const;
        std::vector<std::string>& getBody() { return body; }
        int getNumLines() const { return body.size(); }
        json& getData() { return data; }
    protected:
        std::vector<std::string> header, body;
        json data;
        std::stringstream oss;
        std::locale loc;
    };
}
#endif