#pragma once

#include <string>
#include "common/Colors.h"
#include <sstream>
#include "ReportBase.h"

namespace platform {
    const int MAXL = 133;
    class ReportConsole : public ReportBase {
    public:
        explicit ReportConsole(json data_, bool compare = false, int index = -1) : ReportBase(data_, compare), selectedIndex(index) {};
        virtual ~ReportConsole() = default;
        std::string fileReport();
    private:
        int selectedIndex;
        std::string headerLine(const std::string& text, int utf);
        void header() override;
        void do_header();
        void body() override;
        void do_body();
        void footer(double totalScore);
        void showSummary() override;
        std::stringstream sheader;
        std::stringstream sbody;
    };
};
