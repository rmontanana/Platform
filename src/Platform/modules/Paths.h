#ifndef PATHS_H
#define PATHS_H
#include <string>
#include <filesystem>
#include "DotEnv.h"
namespace platform {
    class Paths {
    public:
        static std::string results() { return "results/"; }
        static std::string hiddenResults() { return "hidden_results/"; }
        static std::string excel() { return "excel/"; }
        static std::string grid() { return "grid/"; }
        static std::string datasets()
        {
            auto env = platform::DotEnv();
            return env.get("source_data");
        }
        static void createPath(const std::string& path)
        {
            // Create directory if it does not exist
            try {
                std::filesystem::create_directory(path);
            }
            catch (std::exception& e) {
                throw std::runtime_error("Could not create directory " + path);
            }
        }
        static std::string excelResults() { return "some_results.xlsx"; }
        static std::string grid_input(const std::string& model)
        {
            return grid() + "grid_" + model + "_input.json";
        }
        static std::string grid_output(const std::string& model)
        {
            return grid() + "grid_" + model + "_output.json";
        }
    };
}
#endif