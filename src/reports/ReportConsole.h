#ifndef REPORTCONSOLE_H
#define REPORTCONSOLE_H
#include <string>
#include "ReportBase.h"
#include "Colors.h"

namespace platform {
    const int MAXL = 133;
    class ReportConsole : public ReportBase {
    public:
        explicit ReportConsole(json data_, bool compare = false, int index = -1) : ReportBase(data_, compare), selectedIndex(index) {};
        virtual ~ReportConsole() = default;
    private:
        int selectedIndex;
        std::string headerLine(const std::string& text, int utf);
        void header() override;
        void body() override;
        void footer(double totalScore);
        void showSummary() override;
    };
};
#endif