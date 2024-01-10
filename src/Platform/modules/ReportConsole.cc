#include <iostream>
#include <sstream>
#include <locale>
#include "ReportConsole.h"
#include "BestScore.h"
#include "CLocale.h"

namespace platform {
    std::string ReportConsole::headerLine(const std::string& text, int utf = 0)
    {
        int n = MAXL - text.length() - 3;
        n = n < 0 ? 0 : n;
        return "* " + text + std::string(n + utf, ' ') + "*\n";
    }

    void ReportConsole::header()
    {
        std::stringstream oss;
        std::cout << Colors::MAGENTA() << std::string(MAXL, '*') << std::endl;
        std::cout << headerLine(
            "Report " + data["model"].get<std::string>() + " ver. " + data["version"].get<std::string>()
            + " with " + std::to_string(data["folds"].get<int>()) + " Folds cross validation and " + std::to_string(data["seeds"].size())
            + " random seeds. " + data["date"].get<std::string>() + " " + data["time"].get<std::string>()
        );
        std::cout << headerLine(data["title"].get<std::string>());
        std::cout << headerLine("Random seeds: " + fromVector("seeds") + " Stratified: " + (data["stratified"].get<bool>() ? "True" : "False"));
        oss << "Execution took  " << std::setprecision(2) << std::fixed << data["duration"].get<float>()
            << " seconds,   " << data["duration"].get<float>() / 3600 << " hours, on " << data["platform"].get<std::string>();
        std::cout << headerLine(oss.str());
        std::cout << headerLine("Score is " + data["score_name"].get<std::string>());
        std::cout << std::string(MAXL, '*') << std::endl;
        std::cout << std::endl;
    }
    void ReportConsole::body()
    {
        auto tmp = ConfigLocale();
        int maxHyper = 15;
        int maxDataset = 7;
        for (const auto& r : data["results"]) {
            maxHyper = std::max(maxHyper, (int)r["hyperparameters"].dump().size());
            maxDataset = std::max(maxDataset, (int)r["dataset"].get<std::string>().size());

        }
        std::cout << Colors::GREEN() << " #  " << std::setw(maxDataset) << std::left << "Dataset" << " Sampl. Feat. Cls Nodes     Edges     States    Score           Time                Hyperparameters" << std::endl;
        std::cout << "=== " << std::string(maxDataset, '=') << " ====== ===== === ========= ========= ========= =============== =================== " << std::string(maxHyper, '=') << std::endl;
        json lastResult;
        double totalScore = 0.0;
        bool odd = true;
        int index = 0;
        for (const auto& r : data["results"]) {
            if (selectedIndex != -1 && index != selectedIndex) {
                index++;
                continue;
            }
            auto color = odd ? Colors::CYAN() : Colors::BLUE();
            std::cout << color;
            std::cout << std::setw(3) << std::right << index++ << " ";
            std::cout << std::setw(maxDataset) << std::left << r["dataset"].get<std::string>() << " ";
            std::cout << std::setw(6) << std::right << r["samples"].get<int>() << " ";
            std::cout << std::setw(5) << std::right << r["features"].get<int>() << " ";
            std::cout << std::setw(3) << std::right << r["classes"].get<int>() << " ";
            std::cout << std::setw(9) << std::setprecision(2) << std::fixed << r["nodes"].get<float>() << " ";
            std::cout << std::setw(9) << std::setprecision(2) << std::fixed << r["leaves"].get<float>() << " ";
            std::cout << std::setw(9) << std::setprecision(2) << std::fixed << r["depth"].get<float>() << " ";
            std::cout << std::setw(8) << std::right << std::setprecision(6) << std::fixed << r["score"].get<double>() << "±" << std::setw(6) << std::setprecision(4) << std::fixed << r["score_std"].get<double>();
            const std::string status = compareResult(r["dataset"].get<std::string>(), r["score"].get<double>());
            std::cout << status;
            std::cout << std::setw(12) << std::right << std::setprecision(6) << std::fixed << r["time"].get<double>() << "±" << std::setw(6) << std::setprecision(4) << std::fixed << r["time_std"].get<double>() << " ";
            std::cout << r["hyperparameters"].dump();
            std::cout << std::endl;
            std::cout << std::flush;
            lastResult = r;
            totalScore += r["score"].get<double>();
            odd = !odd;
        }
        if (data["results"].size() == 1 || selectedIndex != -1) {
            std::cout << std::string(MAXL, '*') << std::endl;
            std::cout << headerLine(fVector("Train scores: ", lastResult["scores_train"], 14, 12));
            std::cout << headerLine(fVector("Test  scores: ", lastResult["scores_test"], 14, 12));
            std::cout << headerLine(fVector("Train  times: ", lastResult["times_train"], 10, 3));
            std::cout << headerLine(fVector("Test   times: ", lastResult["times_test"], 10, 3));
            std::cout << std::string(MAXL, '*') << std::endl;
        } else {
            footer(totalScore);
        }
    }
    void ReportConsole::showSummary()
    {
        for (const auto& item : summary) {
            std::stringstream oss;
            oss << std::setw(3) << std::left << item.first;
            oss << std::setw(3) << std::right << item.second << " ";
            oss << std::left << meaning.at(item.first);
            std::cout << headerLine(oss.str(), 2);
        }
    }

    void ReportConsole::footer(double totalScore)
    {
        std::cout << Colors::MAGENTA() << std::string(MAXL, '*') << std::endl;
        showSummary();
        auto score = data["score_name"].get<std::string>();
        auto best = BestScore::getScore(score);
        if (best.first != "") {
            std::stringstream oss;
            oss << score << " compared to " << best.first << " .:  " << totalScore / best.second;
            std::cout << headerLine(oss.str());
        }
        if (!getExistBestFile() && compare) {
            std::cout << headerLine("*** Best Results File not found. Couldn't compare any result!");
        }
        std::cout << std::string(MAXL, '*') << std::endl << Colors::RESET();
    }
}