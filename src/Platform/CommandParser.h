#ifndef COMMAND_PARSER_H
#define COMMAND_PARSER_H
#include <string>
#include <vector>
#include <tuple>

namespace platform {
    class CommandParser {
    public:
        CommandParser() = default;
        std::pair<char, int> parse(const std::string& color, const std::vector<std::tuple<std::string, char, bool>>& options, const char defaultCommand, const int maxIndex);
        char getCommand() const { return command; };
        int getIndex() const { return index; };
    private:
        void messageError(const std::string& message);
        char command;
        int index;
    };
} /* namespace platform */
#endif /* COMMAND_PARSER_H */