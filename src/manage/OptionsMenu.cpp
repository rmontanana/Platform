#include "OptionsMenu.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include "common/Utils.h"

namespace platform {
    std::string OptionsMenu::to_string()
    {
        bool first = true;
        size_t size = 0;
        std::string result = color_normal + "Options: (";
        for (auto& option : options) {
            if (!first) {
                result += ", ";
                size += 2;
            }
            std::string title = std::get<0>(option);
            auto pos = title.find(std::get<1>(option));
            result += color_normal + title.substr(0, pos) + color_bold + title.substr(pos, 1) + color_normal + title.substr(pos + 1);
            size += title.size();
            first = false;
        }
        if (size + 3 > cols) { // 3 is the size of the "): " at the end
            result = "";
            first = true;
            for (auto& option : options) {
                if (!first) {
                    result += color_normal + ", ";
                }
                result += color_bold + std::get<1>(option);
                first = false;
            }
        }
        result += "): ";
        return result;
    }
    std::tuple<char, int, bool> OptionsMenu::parse(char defaultCommand, int minIndex, int maxIndex)
    {
        bool finished = false;
        while (!finished) {
            std::cout << to_string();
            std::string line;
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
                    return { ' ', -1, true };
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