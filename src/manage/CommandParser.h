#ifndef COMMAND_PARSER_H
#define COMMAND_PARSER_H
#include <string>
#include <vector>
#include <tuple>

namespace platform {
    class CommandParser {
    public:
        CommandParser() = default;
        std::tuple<char, int, bool> parse(const std::string& color, const std::vector<std::tuple<std::string, char, bool>>& options, const char defaultCommand, const int minIndex, const int maxIndex);
        char getCommand() const { return command; };
        int getIndex() const { return index; };
        std::string getErrorMessage() const { return errorMessage; };
    private:
        std::string errorMessage;
        char command;
        int index;
    };
} /* namespace platform */
#endif