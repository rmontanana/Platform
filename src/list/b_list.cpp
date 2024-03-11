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
#include "DatasetsExcel.h"
#include "ResultsDataset.h"
#include "ResultsDatasetExcel.h"
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
    auto excel = program.get<bool>("excel");
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
    auto dataset = program.get<string>("dataset");
    auto score = program.get<string>("score");
    auto model = program.get<string>("model");
    auto excel = program.get<bool>("excel");
    auto results = platform::ResultsDataset(dataset, model, score);
    results.load();
    results.sortModel();
    if (results.empty()) {
        std::cerr << Colors::RED() << "No results found for dataset " << dataset << " and model " << model << Colors::RESET() << std::endl;
        exit(1);
    }
    int maxModel = results.maxModelSize();
    int maxHyper = results.maxHyperSize();
    double maxResult = results.maxResultScore();
    // Build data for the Report
    json data = json::object();
    data["results"] = json::array();
    data["max_models"] = json::object(); // Max score per model
    for (const auto& result : results) {
        auto results = result.getData();
        if (!data["max_models"].contains(result.getModel())) {
            data["max_models"][result.getModel()] = 0;
        }
        for (const auto& item : results["results"]) {
            if (item["dataset"] == dataset) {

                // Store data for Excel report
                json res = json::object();
                res["date"] = result.getDate();
                res["time"] = result.getTime();
                res["model"] = result.getModel();
                res["score"] = item["score"].get<double>();
                res["hyperparameters"] = item["hyperparameters"].dump();
                data["results"].push_back(res);
                if (item["score"].get<double>() > data["max_models"][result.getModel()]) {
                    data["max_models"][result.getModel()] = item["score"].get<double>();
                }
                break;
            }
        }
    }
    //
    // List the results
    //
    std::cout << Colors::GREEN() << "Results of dataset " << dataset << " - for " << model << " model" << std::endl;
    std::cout << "There are " << results.size() << " results" << std::endl;
    std::cout << Colors::GREEN() << " #  " << std::setw(maxModel + 1) << std::left << "Model" << "Date       Time     Score       Hyperparameters" << std::endl;
    std::cout << "=== " << std::string(maxModel, '=') << " ========== ======== =========== " << std::string(maxHyper, '=') << std::endl;
    auto i = 0;
    for (const auto& item : data["results"]) {
        auto color = (i % 2) ? Colors::BLUE() : Colors::CYAN();
        auto score = item["score"].get<double>();
        color = score == data["max_models"][item["model"].get<std::string>()] ? Colors::YELLOW() : color;
        color = score == maxResult ? Colors::RED() : color;
        std::cout << color << std::setw(3) << std::fixed << std::right << i++ << " ";
        std::cout << std::setw(maxModel) << std::left << item["model"].get<std::string>() << " ";
        std::cout << color << item["date"].get<std::string>() << " ";
        std::cout << color << item["time"].get<std::string>() << " ";
        std::cout << std::setw(11) << std::setprecision(9) << std::fixed << score << " ";
        std::cout << item["hyperparameters"].get<std::string>() << std::endl;
    }
    if (excel) {
        data["dataset"] = dataset;
        data["score"] = score;
        data["model"] = model;
        data["lengths"]["maxModel"] = maxModel;
        data["lengths"]["maxHyper"] = maxHyper;
        data["maxResult"] = maxResult;
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