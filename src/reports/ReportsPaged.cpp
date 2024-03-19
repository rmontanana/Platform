#include "common/Colors.h"
#include "ReportsPaged.h"

namespace platform {
    ReportsPaged::ReportsPaged()
    {
        loc = std::locale("es_ES.UTF-8");
        oss.imbue(loc);
    }
    std::string ReportsPaged::getOutput() const
    {
        std::string s;
        for (const auto& piece : header) s += piece;
        for (const auto& piece : body) s += piece;
        return s;
    }
    std::string ReportsPaged::getHeader() const
    {
        std::string s;
        for (const auto& piece : header) s += piece;
        return s;
    }
}
