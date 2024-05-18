#ifndef COLORS_H
#define COLORS_H
#include <string>
class Colors {
public:
    static std::string BLACK() { return "\033[1;30m"; }
    static std::string IBLACK() { return "\033[0;90m"; }
    static std::string BLUE() { return "\033[1;34m"; }
    static std::string IBLUE() { return "\033[0;94m"; }
    static std::string CYAN() { return "\033[1;36m"; }
    static std::string ICYAN() { return "\033[0;96m"; }
    static std::string GREEN() { return "\033[1;32m"; }
    static std::string IGREEN() { return "\033[0;92m"; }
    static std::string MAGENTA() { return "\033[1;35m"; }
    static std::string IMAGENTA() { return "\033[0;95m"; }
    static std::string RED() { return "\033[1;31m"; }
    static std::string IRED() { return "\033[0;91m"; }
    static std::string YELLOW() { return "\033[1;33m"; }
    static std::string IYELLOW() { return "\033[0;93m"; }
    static std::string WHITE() { return "\033[1;37m"; }
    static std::string IWHITE() { return "\033[0;97m"; }
    static std::string RESET() { return "\033[0m"; }
    static std::string BOLD() { return "\033[1m"; }
    static std::string UNDERLINE() { return "\033[4m"; }
    static std::string BLINK() { return "\033[5m"; }
    static std::string REVERSE() { return "\033[7m"; }
    static std::string CONCEALED() { return "\033[8m"; }
    static std::string CLRSCR() { return "\033[2J\033[1;1H"; }
};
#endif