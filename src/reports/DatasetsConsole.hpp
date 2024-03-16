#pragma once

#include <locale>
#include <map>
#include <sstream>
#include <nlohmann/json.hpp>
#include "common/Colors.h"
#include "common/Datasets.h"

namespace platform {
    const int BALANCE_LENGTH = 75;
    struct separated_datasets : numpunct<char> {
        char do_decimal_point() const { return ','; }
        char do_thousands_sep() const { return '.'; }
        std::string do_grouping() const { return "\03"; }
    };

    class DatasetsConsole {
    public:
        DatasetsConsole() = default;
        ~DatasetsConsole() = default;
        std::string getOutput() const { return output.str(); }
        int getNumLines() const { return numLines; }
        json& getData() { return data; }
        std::string outputBalance(const std::string& balance)
        {
            auto temp = std::string(balance);
            while (temp.size() > BALANCE_LENGTH - 1) {
                auto part = temp.substr(0, BALANCE_LENGTH);
                output << part << std::endl;
                output << setw(52) << " ";
                temp = temp.substr(BALANCE_LENGTH);
            }
            return temp;
        }
        void list_datasets()
        {
            auto datasets = platform::Datasets(false, platform::Paths::datasets());
            locale mylocale(std::cout.getloc(), new separated_datasets);
            locale::global(mylocale);
            output.imbue(mylocale);
            output << Colors::GREEN() << " #  Dataset                        Sampl. Feat. Cls Balance" << std::endl;
            std::string balanceBars = std::string(BALANCE_LENGTH, '=');
            output << "=== ============================== ====== ===== === " << balanceBars << std::endl;
            int num = 0;
            for (const auto& dataset : datasets.getNames()) {
                auto color = num % 2 ? Colors::CYAN() : Colors::BLUE();
                output << color << setw(3) << right << num++ << " ";
                output << setw(30) << left << dataset << " ";
                datasets.loadDataset(dataset);
                auto nSamples = datasets.getNSamples(dataset);
                output << setw(6) << right << nSamples << " ";
                output << setw(5) << right << datasets.getFeatures(dataset).size() << " ";
                output << setw(3) << right << datasets.getNClasses(dataset) << " ";
                std::stringstream oss;
                std::string sep = "";
                for (auto number : datasets.getClassesCounts(dataset)) {
                    oss << sep << std::setprecision(2) << fixed << (float)number / nSamples * 100.0 << "% (" << number << ")";
                    sep = " / ";
                }
                auto balance = outputBalance(oss.str());
                output << balance << std::endl;
                // Store data for Excel report
                data[dataset] = json::object();
                data[dataset]["samples"] = nSamples;
                data[dataset]["features"] = datasets.getFeatures(dataset).size();
                data[dataset]["classes"] = datasets.getNClasses(dataset);
                data[dataset]["balance"] = oss.str();
            }
            numLines = num + 2;
        }
    private:
        std::stringstream output;
        json data;
        int numLines = 0;
    };
}

