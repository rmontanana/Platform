#ifndef COLORS_H
#define COLORS_H
class Colors {
public:
    static std::string MAGENTA() { return "\033[1;35m"; }
    static std::string BLUE() { return "\033[1;34m"; }
    static std::string CYAN() { return "\033[1;36m"; }
    static std::string GREEN() { return "\033[1;32m"; }
    static std::string YELLOW() { return "\033[1;33m"; }
    static std::string RED() { return "\033[1;31m"; }
    static std::string WHITE() { return "\033[1;37m"; }
    static std::string IBLUE() { return "\033[0;94m"; }
    static std::string RESET() { return "\033[0m"; }
};
#endif // COLORS_H