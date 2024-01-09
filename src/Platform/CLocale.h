#ifndef LOCALE_H
#define LOCALE_H
#include <locale>
#include <iostream>
#include <string>
namespace platform {
    struct separation : std::numpunct<char> {
        char do_decimal_point() const { return ','; }
        char do_thousands_sep() const { return '.'; }
        std::string do_grouping() const { return "\03"; }
    };
    class ConfigLocale {
    public:
        explicit ConfigLocale()
        {
            std::locale mylocale(std::cout.getloc(), new separation);
            std::locale::global(mylocale);
            std::cout.imbue(mylocale);
        }
    };
}
#endif 