#ifndef OPTIONS_MENU_H
#define OPTIONS_MENU_H
#include <string>
#include <vector>
#include <tuple>

namespace platform {
    class OptionsMenu {
    public:
        OptionsMenu(std::vector<std::tuple<std::string, char, bool>>& options, std::string color_normal, std::string color_bold, int cols) : options(options), color_normal(color_normal), color_bold(color_bold), cols(cols) {}
        std::string to_string();
        std::tuple<char, int, bool> parse(char defaultCommand, int minIndex, int maxIndex);
        char getCommand() const { return command; };
        int getIndex() const { return index; };
        std::string getErrorMessage() const { return errorMessage; };
    private:
        std::vector<std::tuple<std::string, char, bool>>& options;
        std::string color_normal, color_bold;
        int cols;
        std::string errorMessage;
        char command;
        int index;
    };
} /* namespace platform */
#endif