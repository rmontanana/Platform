#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "common/Paths.h"
#include "results/JsonValidator.h"
#include "results/SchemaV1_0.h"
#include "config_platform.h"

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
    argparse::ArgumentParser program("b_results", { platform_project_version.begin(), platform_project_version.end() });
    program.add_description("Check the results files and optionally fixes them.");
    program.add_argument("--fix").help("Fix any errors in results").default_value(false).implicit_value(true);
    program.add_argument("--file").help("check only this results file").default_value("");
    std::string nameSuffix = "results_";
    std::string schemaVersion = "1.0";
    bool fix_it = false;
    std::string selected_file;
    try {
        program.parse_args(argc, argv);
        fix_it = program.get<bool>("fix");
        selected_file = program.get<std::string>("file");
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        exit(1);
    }
    //
    // Determine the files to process
    //
    std::vector<std::string> result_files;
    int max_length = 0;
    if (selected_file != "") {
        if (!selected_file.starts_with(platform::Paths::results())) {
            selected_file = platform::Paths::results() + selected_file;
        }
        // Only check the selected file
        result_files.push_back(selected_file);
        max_length = selected_file.length();
    } else {
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
    }
    //
    // Process the results files
    //
    if (result_files.empty()) {
        std::cerr << "Error: No result files found." << std::endl;
        return 1;
    }
    std::string header_message = "Processing " + std::to_string(result_files.size()) + " result files.";
    header(header_message, max_length, "*");
    platform::JsonValidator validator(platform::SchemaV1_0::schema);
    int n_errors = 0;
    std::vector<std::string> files_with_errors;
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
                std::cout << " -> File fixed." << std::endl;
            }
            files_with_errors.push_back(file_name);
        }
    }
    if (n_errors == 0) {
        header("All files are valid.", max_length, "*");
    } else {
        std::string $verb = (fix_it) ? "had" : "have";
        std::string msg = std::to_string(n_errors) + " files " + $verb + " errors.";
        header(msg, max_length, "*");
        for (const auto& file_name : files_with_errors) {
            std::cout << "- " << file_name << std::endl;
        }
    }
    return 0;
}
