#ifndef SYMBOLS_H
#define SYMBOLS_H
#include <string>
namespace platform {
    class Symbols {
    public:
        inline static const std::string check_mark{ "\u2714" };
        inline static const std::string exclamation{ "\u2757" };
        inline static const std::string black_star{ "\u2605" };
        inline static const std::string cross{ "\u2717" };
        inline static const std::string upward_arrow{ "\u27B6" };
        inline static const std::string downward_arrow{ "\u27B4" };
        inline static const std::string up_arrow{ "\u2B06" };
        inline static const std::string down_arrow{ "\u2B07" };
        inline static const std::string ellipsis{ "\u2026" };
        inline static const std::string equal_best{ check_mark };
        inline static const std::string better_best{ black_star };
        inline static const std::string notebook{ "\U0001F5C8" };
    };
}
#endif