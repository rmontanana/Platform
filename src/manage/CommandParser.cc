#include "CommandParser.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include "Colors.h"
#include "Utils.h"

namespace platform {
    void CommandParser::messageError(const std::string& message)
    {
        std::cout << Colors::RED() << message << Colors::RESET() << std::endl;
    }
    std::pair<char, int> CommandParser::parse(const std::string& color, const std::vector<std::tuple<std::string, char, bool>>& options, const char defaultCommand, const int maxIndex)
    {
        bool finished = false;
        while (!finished) {
            std::stringstream oss;
            std::string line;
            oss << color << "Choose option (";
            bool first = true;
            for (auto& option : options) {
                if (first) {
                    first = false;
                } else {
                    oss << ", ";
                }
                oss << std::get<char>(option) << "=" << std::get<std::string>(option);
            }
            oss << "): ";
            std::cout << oss.str();
            getline(std::cin, line);
            std::cout << Colors::RESET();
            line = trim(line);
            if (line.size() == 0)
                continue;
            if (all_of(line.begin(), line.end(), ::isdigit)) {
                command = defaultCommand;
                index = stoi(line);
                if (index > maxIndex || index < 0) {
                    messageError("Index out of range");
                    continue;
                }
                finished = true;
                break;
            }
            bool found = false;
            for (auto& option : options) {
                if (line[0] == std::get<char>(option)) {
                    found = true;
                    // it's a match
                    line.erase(line.begin());
                    line = trim(line);
                    if (std::get<bool>(option)) {
                        // The option requires a value
                        if (line.size() == 0) {
                            messageError("Option " + std::get<std::string>(option) + " requires a value");
                            break;
                        }
                        try {
                            index = stoi(line);
                            if (index > maxIndex || index < 0) {
                                messageError("Index out of range");
                                break;
                            }
                        }
                        catch (const std::invalid_argument& ia) {
                            messageError("Invalid value: " + line);
                            break;
                        }
                    } else {
                        if (line.size() > 0) {
                            messageError("option " + std::get<std::string>(option) + " doesn't accept values");
                            break;
                        }
                    }
                    command = std::get<char>(option);
                    finished = true;
                    break;
                }
            }
            if (!found) {
                messageError("I don't know " + line);
            }
        }
        return { command, index };
    }
} /* namespace platform */