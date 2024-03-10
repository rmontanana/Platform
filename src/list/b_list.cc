#include <iostream>
#include <locale>
#include <map>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "common/Paths.h"
#include "common/Colors.h"
#include "common/Datasets.h"
#include "DatasetsExcel.h"
#include "config.h"

const int BALANCE_LENGTH = 75;

struct separated : numpunct<char> {
    char do_decimal_point() const { return ','; }
    char do_thousands_sep() const { return '.'; }
    std::string do_grouping() const { return "\03"; }
};

std::string outputBalance(const std::string& balance)
{
    auto temp = std::string(balance);
    while (temp.size() > BALANCE_LENGTH - 1) {
        auto part = temp.substr(0, BALANCE_LENGTH);
        std::cout << part << std::endl;
        std::cout << setw(52) << " ";
        temp = temp.substr(BALANCE_LENGTH);
    }
    return temp;
}

void list_datasets(argparse::ArgumentParser& program)
{
    auto datasets = platform::Datasets(false, platform::Paths::datasets());
    auto excel = program.get<bool>("--excel");
    locale mylocale(std::cout.getloc(), new separated);
    locale::global(mylocale);
    std::cout.imbue(mylocale);
    std::cout << Colors::GREEN() << " #  Dataset                        Sampl. Feat. Cls Balance" << std::endl;
    std::string balanceBars = std::string(BALANCE_LENGTH, '=');
    std::cout << "=== ============================== ====== ===== === " << balanceBars << std::endl;
    int num = 0;
    json data;
    for (const auto& dataset : datasets.getNames()) {
        auto color = num % 2 ? Colors::CYAN() : Colors::BLUE();
        std::cout << color << setw(3) << right << num++ << " ";
        std::cout << setw(30) << left << dataset << " ";
        datasets.loadDataset(dataset);
        auto nSamples = datasets.getNSamples(dataset);
        std::cout << setw(6) << right << nSamples << " ";
        std::cout << setw(5) << right << datasets.getFeatures(dataset).size() << " ";
        std::cout << setw(3) << right << datasets.getNClasses(dataset) << " ";
        std::stringstream oss;
        std::string sep = "";
        for (auto number : datasets.getClassesCounts(dataset)) {
            oss << sep << std::setprecision(2) << fixed << (float)number / nSamples * 100.0 << "% (" << number << ")";
            sep = " / ";
        }
        auto balance = outputBalance(oss.str());
        std::cout << balance << std::endl;
        // Store data for Excel report
        data[dataset] = json::object();
        data[dataset]["samples"] = nSamples;
        data[dataset]["features"] = datasets.getFeatures(dataset).size();
        data[dataset]["classes"] = datasets.getNClasses(dataset);
        data[dataset]["balance"] = oss.str();
    }
    if (excel) {
        auto report = platform::DatasetsExcel();
        report.report(data);
        std::cout << std::endl << Colors::GREEN() << "Output saved in " << report.getFileName() << std::endl;
    }
}

void list_results(argparse::ArgumentParser& program)
{
    std::cout << "Results" << std::endl;
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("b_list", { platform_project_version.begin(), platform_project_version.end() });
    //
    // datasets subparser
    //
    argparse::ArgumentParser datasets_command("datasets");
    datasets_command.add_description("List datasets available in the platform.");
    datasets_command.add_argument("--excel")
        .help("Output in Excel format")
        .default_value(false)
        .implicit_value(true);
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