#ifndef UTILS_H
#define UTILS_H

#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>

extern char **environ;

namespace platform {
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
    static std::vector<std::string> split(const std::string& text, char delimiter)
    {
        std::vector<std::string> result;
        std::stringstream ss(text);
        std::string token;
        while (std::getline(ss, token, delimiter)) {
            result.push_back(trim(token));
        }
        return result;
    }
    inline double compute_std(std::vector<double> values, double mean)
    {
        // Compute standard devation of the values
        double sum = 0.0;
        for (const auto& value : values) {
            sum += std::pow(value - mean, 2);
        }
        double variance = sum / values.size();
        return std::sqrt(variance);
    }
    inline std::string get_date()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%Y-%m-%d");
        return oss.str();
    }
    inline std::string get_time()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%H:%M:%S");
        return oss.str();
    }
    static void openFile(const std::string& fileName)
    {
        // #ifdef __APPLE__
        //     // macOS uses the "open" command
        //     std::string command = "open";
        // #elif defined(__linux__)
        //     // Linux typically uses "xdg-open"
        //     std::string command = "xdg-open";
        // #else
        //     // For other OSes, do nothing or handle differently
        //     std::cerr << "Unsupported platform." << std::endl;
        //     return;
        // #endif
        //     execlp(command.c_str(), command.c_str(), fileName.c_str(), NULL);
#ifdef __APPLE__
        const char* tool = "/usr/bin/open";
#elif defined(__linux__)
        const char* tool = "/usr/bin/xdg-open";
#else
        std::cerr << "Unsupported platform." << std::endl;
        return;
#endif

        // We'll build an argv array for execve:
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(tool));               // argv[0]
        argv.push_back(const_cast<char*>(fileName.c_str()));   // argv[1]
        argv.push_back(nullptr);

        // Make a new environment array, skipping BASH_FUNC_ variables
        std::vector<std::string> filteredEnv;
        for (char** env = environ; *env != nullptr; ++env) {
            // *env is a string like "NAME=VALUE"
            // We want to skip those starting with "BASH_FUNC_"
            if (strncmp(*env, "BASH_FUNC_", 10) == 0) {
                // skip it
                continue;
            }
            filteredEnv.push_back(*env);
        }

        // Convert filteredEnv into a char* array
        std::vector<char*> envp;
        for (auto& var : filteredEnv) {
            envp.push_back(const_cast<char*>(var.c_str()));
        }
        envp.push_back(nullptr);

        // Now call execve with the cleaned environment
        // NOTE: You may need a full path to the tool if it's not in PATH, or use which() logic
        // For now, let's assume "open" or "xdg-open" is found in the default PATH:
        execve(tool, argv.data(), envp.data());

        // If we reach here, execve failed
        perror("execve failed");
        // This would terminate your current process if it's not in a child
        // Usually you'd do something like:
        _exit(EXIT_FAILURE);
    }
}
#endif