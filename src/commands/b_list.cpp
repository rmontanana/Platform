#include <iostream>
#include <locale>
#include <map>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "main/Models.h"
#include "main/modelRegister.h"
#include "common/Paths.h"
#include "common/Colors.h"
#include "common/Datasets.h"
#include "reports/DatasetsExcel.h"
#include "reports/DatasetsConsole.h"
#include "results/ResultsDataset.h"
#include "results/ResultsDatasetExcel.h"
#include "results/ResultsDatasetConsole.h"
#include "config.h"


void list_datasets(argparse::ArgumentParser& program)
{
    auto excel = program.get<bool>("excel");
    auto report = platform::DatasetsConsole();
    report.list_datasets();
    std::cout << report.getOutput();
    if (excel) {
        auto data = report.getData();
        auto report = platform::DatasetsExcel();
        report.report(data);
        std::cout << std::endl << Colors::GREEN() << "Output saved in " << report.getFileName() << std::endl;
    }
}

void list_results(argparse::ArgumentParser& program)
{
    auto dataset = program.get<string>("dataset");
    auto score = program.get<string>("score");
    auto model = program.get<string>("model");
    auto excel = program.get<bool>("excel");
    auto report = platform::ResultsDatasetsConsole();
    report.list_results(dataset, score, model);
    std::cout << report.getOutput();
    if (excel) {
        auto data = report.getData();
        auto report = platform::ResultsDatasetExcel();
        report.report(data);
        std::cout << std::endl << Colors::GREEN() << "Output saved in " << report.getFileName() << std::endl;
    }
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("b_list", { platform_project_version.begin(), platform_project_version.end() });
    //
    // datasets subparser
    //
    argparse::ArgumentParser datasets_command("datasets");
    datasets_command.add_description("List datasets available in the platform.");
    datasets_command.add_argument("--excel").help("Output in Excel format").default_value(false).implicit_value(true);
    //
    // results subparser
    //
    argparse::ArgumentParser results_command("results");
    results_command.add_description("List the results of a given dataset.");
    auto datasets = platform::Datasets(false, platform::Paths::datasets());
    results_command.add_argument("-d", "--dataset")
        .help("Dataset to use " + datasets.toString())
        .required()
        .action([](const std::string& value) {
        auto datasets = platform::Datasets(false, platform::Paths::datasets());
        static const std::vector<std::string> choices = datasets.getNames();
        if (find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw std::runtime_error("Dataset must be one of " + datasets.toString());
            }
    );
    results_command.add_argument("-m", "--model")
        .help("Model to use: " + platform::Models::instance()->toString() + " or any")
        .default_value("any")
        .action([](const std::string& value) {
        std::vector<std::string> valid(platform::Models::instance()->getNames());
        valid.push_back("any");
        static const std::vector<std::string> choices = valid;
        if (find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw std::runtime_error("Model must be one of " + platform::Models::instance()->toString() + " or any");
            }
    );
    results_command.add_argument("--excel").help("Output in Excel format").default_value(false).implicit_value(true);
    results_command.add_argument("-s", "--score").default_value("accuracy").help("Filter results of the score name supplied");

    // Add subparsers
    program.add_subparser(datasets_command);
    program.add_subparser(results_command);
    // Parse command line and execute
    try {
        program.parse_args(argc, argv);
        bool found = false;
        map<std::string, void(*)(argparse::ArgumentParser&)> commands = { {"datasets", &list_datasets}, {"results", &list_results} };
        for (const auto& command : commands) {
            if (program.is_subcommand_used(command.first)) {
                std::invoke(command.second, program.at<argparse::ArgumentParser>(command.first));
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("You must specify one of the following commands: datasets, results\n");
        }
    }
    catch (const exception& err) {
        cerr << err.what() << std::endl;
        cerr << program;
        exit(1);
    }
    std::cout << Colors::RESET() << std::endl;
    return 0;
}