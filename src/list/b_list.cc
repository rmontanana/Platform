#include <iostream>
#include <locale>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "Paths.h"
#include "Colors.h"
#include "Datasets.h"
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

int main(int argc, char** argv)
{
    auto datasets = platform::Datasets(false, platform::Paths::datasets());
    argparse::ArgumentParser program("b_list", { project_version.begin(), project_version.end() });
    program.add_argument("--excel")
        .help("Output in Excel format")
        .default_value(false)
        .implicit_value(true);
    program.parse_args(argc, argv);
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
    std::cout << Colors::RESET() << std::endl;
    if (excel) {
        auto report = platform::DatasetsExcel();
        report.report(data);
        std::cout << "Output saved in " << report.getFileName() << std::endl;
    }
    return 0;
}
