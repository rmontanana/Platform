#ifndef UTILS_H
#define UTILS_H
#include <sstream>
#include <string>
#include <vector>
namespace platform {
    //static std::vector<std::string> split(const std::string& text, char delimiter);
    static std::vector<std::string> split(const std::string& text, char delimiter)
    {
        std::vector<std::string> result;
        std::stringstream ss(text);
        std::string token;
        while (std::getline(ss, token, delimiter)) {
            result.push_back(token);
        }
        return result;
    }
    static std::string trim(const std::string& str)
    {
        std::string result = str;
        result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](int ch) {
            return !std::isspace(ch);
            }));
        result.erase(std::find_if(result.rbegin(), result.rend(), [](int ch) {
            return !std::isspace(ch);
            }).base(), result.end());
        return result;
    }
}
#endif