#include "common/Colors.h"
#include "common/Datasets.h"
#include "common/Paths.h"
#include "DatasetsConsole.h"

namespace platform {
    const int DatasetsConsole::BALANCE_LENGTH = 75;
    std::string DatasetsConsole::outputBalance(const std::string& balance)
    {
        auto temp = std::string(balance);
        while (temp.size() > DatasetsConsole::BALANCE_LENGTH - 1) {
            auto part = temp.substr(0, DatasetsConsole::BALANCE_LENGTH);
            output << part << std::endl;
            output << setw(52) << " ";
            temp = temp.substr(DatasetsConsole::BALANCE_LENGTH);
        }
        return temp;
    }
    void DatasetsConsole::list_datasets()
    {
        output.str("");
        auto datasets = platform::Datasets(false, platform::Paths::datasets());
        auto loc = std::locale("es_ES");
        output.imbue(loc);
        output << Colors::GREEN() << " #  Dataset                        Sampl. Feat. Cls Balance" << std::endl;
        std::string balanceBars = std::string(DatasetsConsole::BALANCE_LENGTH, '=');
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
}

