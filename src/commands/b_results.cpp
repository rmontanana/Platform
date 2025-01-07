#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <regex>
#include <nlohmann/json.hpp>
#include "common/Paths.h"
#include "results/JsonValidator.h"
#include "results/SchemaV1_0.h"

using json = nlohmann::json;
namespace fs = std::filesystem;
void header(const std::string& message, int length, const std::string& symbol)
{
    std::cout << std::string(length + 11, symbol[0]) << std::endl;
    std::cout << symbol << " " << std::setw(length + 7) << std::left << message << " " << symbol << std::endl;
    std::cout << std::string(length + 11, symbol[0]) << std::endl;
}
int main(int argc, char* argv[])
{
    std::string nameSuffix = "results_";
    std::string schemaVersion = "1.0";
    bool fix_it = false;

    std::vector<std::string> result_files;
    int max_length = 0;
    // Load the result files and find the longest file name
    for (const auto& entry : fs::directory_iterator(platform::Paths::results())) {
        if (entry.is_regular_file() && entry.path().filename().string().starts_with(nameSuffix) && entry.path().filename().string().ends_with(".json")) {
            std::string fileName = entry.path().string();
            if (fileName.length() > max_length) {
                max_length = fileName.length();
            }
            result_files.push_back(fileName);
        }
    }
    // Process the result files
    if (result_files.empty()) {
        std::cerr << "Error: No result files found." << std::endl;
        return 1;
    }
    std::string header_message = "Processing " + std::to_string(result_files.size()) + " result files.";
    header(header_message, max_length, "*");
    platform::JsonValidator validator(platform::SchemaV1_0::schema);
    int n_errors = 0;
    for (const auto& file_name : result_files) {
        std::vector<std::string> errors = validator.validate(file_name);
        if (!errors.empty()) {
            n_errors++;
            std::cout << std::setw(max_length) << std::left << file_name << ": " << errors.size() << " Errors:" << std::endl;
            for (const auto& error : errors) {
                std::cout << " - " << error << std::endl;
            }
            if (fix_it) {
                validator.fix_it(file_name);
                std::cout << "  -> File fixed." << std::endl;
            }
        }
    }
    if (n_errors == 0) {
        header("All files are valid.", max_length, "*");
    } else {
        std::string $verb = (fix_it) ? "had" : "have";
        std::string msg = std::to_string(n_errors) + " files " + $verb + " errors.";
        header(msg, max_length, "*");
    }
    return 0;
}
