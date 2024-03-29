#include "common/Colors.h"
#include "common/Datasets.h"
#include "common/Paths.h"
#include "DatasetsConsole.h"

namespace platform {
    const int DatasetsConsole::BALANCE_LENGTH = 75;
    void DatasetsConsole::split_lines(int name_len, std::string line, const std::string& balance)
    {
        auto temp = std::string(balance);
        while (temp.size() > DatasetsConsole::BALANCE_LENGTH - 1) {
            auto part = temp.substr(0, DatasetsConsole::BALANCE_LENGTH);
            line += part + "\n";
            body.push_back(line);
            line = string(name_len + 22, ' ');
            temp = temp.substr(DatasetsConsole::BALANCE_LENGTH);
        }
        line += temp + "\n";
        body.push_back(line);
    }
    void DatasetsConsole::report()
    {
        header.clear();
        body.clear();
        auto datasets = platform::Datasets(false, platform::Paths::datasets());
        std::stringstream sheader;
        auto datasets_names = datasets.getNames();
        int maxName = std::max(size_t(7), (*max_element(datasets_names.begin(), datasets_names.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size());
        std::vector<std::string> header_labels = { " #", "Dataset", "Sampl.", "Feat.", "Cls", "Balance" };
        std::vector<int> header_lengths = { 3, maxName, 6, 5, 3, DatasetsConsole::BALANCE_LENGTH };
        sheader << Colors::GREEN();
        for (int i = 0; i < header_labels.size(); i++) {
            sheader << setw(header_lengths[i]) << left << header_labels[i] << " ";
        }
        sheader << std::endl;
        header.push_back(sheader.str());
        std::string sline;
        for (int i = 0; i < header_labels.size(); i++) {
            sline += std::string(header_lengths[i], '=') + " ";
        }
        sline += "\n";
        header.push_back(sline);
        int num = 0;
        for (const auto& dataset : datasets.getNames()) {
            std::stringstream line;
            line.imbue(loc);
            auto color = num % 2 ? Colors::CYAN() : Colors::BLUE();
            line << color << setw(3) << right << num++ << " ";
            line << setw(maxName) << left << dataset << " ";
            datasets.loadDataset(dataset);
            auto nSamples = datasets.getNSamples(dataset);
            line << setw(6) << right << nSamples << " ";
            line << setw(5) << right << datasets.getFeatures(dataset).size() << " ";
            line << setw(3) << right << datasets.getNClasses(dataset) << " ";
            std::string sep = "";
            oss.str("");
            for (auto number : datasets.getClassesCounts(dataset)) {
                oss << sep << std::setprecision(2) << fixed << (float)number / nSamples * 100.0 << "% (" << number << ")";
                sep = " / ";
            }
            split_lines(maxName, line.str(), oss.str());
            // Store data for Excel report
            data[dataset] = json::object();
            data[dataset]["samples"] = nSamples;
            data[dataset]["features"] = datasets.getFeatures(dataset).size();
            data[dataset]["classes"] = datasets.getNClasses(dataset);
            data[dataset]["balance"] = oss.str();
        }
    }
}

