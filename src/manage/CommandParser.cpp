#include "CommandParser.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include "common/Colors.h"
#include "common/Utils.h"

namespace platform {

    std::tuple<char, int, bool> CommandParser::parse(const std::string& color, const std::vector<std::tuple<std::string, char, bool>>& options, const char defaultCommand, const int minIndex, const int maxIndex)
    {
        bool finished = false;
        while (!finished) {
            std::stringstream oss;
            std::string line;
            oss << color << "Options (";
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
            line = trim(line);
            if (line.size() == 0) {
                errorMessage = "No command";
                return { defaultCommand, 0, true };
            }
            if (all_of(line.begin(), line.end(), ::isdigit)) {
                command = defaultCommand;
                index = stoi(line);
                if (index > maxIndex || index < minIndex) {
                    errorMessage = "Index out of range";
                    return { command, index, true };
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
                            errorMessage = "Option " + std::get<std::string>(option) + " requires a value";
                            return { command, index, true };
                        }
                        try {
                            index = stoi(line);
                            if (index > maxIndex || index < 0) {
                                errorMessage = "Index out of range";
                                return { command, index, true };
                            }
                        }
                        catch (const std::invalid_argument& ia) {
                            errorMessage = "Invalid value: " + line;
                            return { command, index, true };
                        }
                    } else {
                        if (line.size() > 0) {
                            errorMessage = "option " + std::get<std::string>(option) + " doesn't accept values";
                            return { command, index, true };
                        }
                    }
                    command = std::get<char>(option);
                    finished = true;
                    break;
                }
            }
            if (!found) {
                errorMessage = "I don't know " + line;
                return { command, index, true };
            }
        }
        return { command, index, false };
    }
} /* namespace platform */