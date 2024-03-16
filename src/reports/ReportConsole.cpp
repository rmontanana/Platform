#include <iostream>
#include <locale>
#include "best/BestScore.h"
#include "common/CLocale.h"
#include "ReportConsole.h"

namespace platform {
    std::string ReportConsole::headerLine(const std::string& text, int utf = 0)
    {
        int n = MAXL - text.length() - 3;
        n = n < 0 ? 0 : n;
        return "* " + text + std::string(n + utf, ' ') + "*\n";
    }
    void ReportConsole::do_header()
    {
        sheader.str("");
        std::stringstream oss;
        sheader << Colors::MAGENTA() << std::string(MAXL, '*') << std::endl;
        sheader << headerLine(
            "Report " + data["model"].get<std::string>() + " ver. " + data["version"].get<std::string>()
            + " with " + std::to_string(data["folds"].get<int>()) + " Folds cross validation and " + std::to_string(data["seeds"].size())
            + " random seeds. " + data["date"].get<std::string>() + " " + data["time"].get<std::string>()
        );
        sheader << headerLine(data["title"].get<std::string>());
        sheader << headerLine(
            "Random seeds: " + fromVector("seeds") + " Discretized: " + (data["discretized"].get<bool>() ? "True" : "False")
            + " Stratified: " + (data["stratified"].get<bool>() ? "True" : "False")
        );
        oss << "Execution took  " << std::setprecision(2) << std::fixed << data["duration"].get<float>()
            << " seconds,   " << data["duration"].get<float>() / 3600 << " hours, on " << data["platform"].get<std::string>();
        sheader << headerLine(oss.str());
        sheader << headerLine("Score is " + data["score_name"].get<std::string>());
        sheader << std::string(MAXL, '*') << std::endl;
        sheader << std::endl;
    }
    void ReportConsole::header()
    {
        std::cout << sheader.str();
    }
    std::string ReportConsole::fileReport()
    {
        do_header();
        do_body();
        std::stringstream oss;
        oss << sheader.str() << sbody.str();
        return oss.str();
    }
    void ReportConsole::do_body()
    {
        sbody.str("");
        auto tmp = ConfigLocale();
        int maxHyper = 15;
        int maxDataset = 7;
        for (const auto& r : data["results"]) {
            maxHyper = std::max(maxHyper, (int)r["hyperparameters"].dump().size());
            maxDataset = std::max(maxDataset, (int)r["dataset"].get<std::string>().size());

        }
        sbody << Colors::GREEN() << " #  " << std::setw(maxDataset) << std::left << "Dataset" << " Sampl. Feat. Cls Nodes     Edges     States    Score           Time                Hyperparameters" << std::endl;
        sbody << "=== " << std::string(maxDataset, '=') << " ====== ===== === ========= ========= ========= =============== =================== " << std::string(maxHyper, '=') << std::endl;
        json lastResult;
        double totalScore = 0.0;
        int index = 0;
        for (const auto& r : data["results"]) {
            if (selectedIndex != -1 && index != selectedIndex) {
                index++;
                continue;
            }
            auto color = (index % 2) ? Colors::CYAN() : Colors::BLUE();
            sbody << color;
            std::string separator{ " " };
            if (r.find("notes") != r.end()) {
                separator = r["notes"].size() > 0 ? Colors::YELLOW() + Symbols::notebook + color : " ";
            }
            sbody << std::setw(3) << std::right << index++ << separator;
            sbody << std::setw(maxDataset) << std::left << r["dataset"].get<std::string>() << " ";
            sbody << std::setw(6) << std::right << r["samples"].get<int>() << " ";
            sbody << std::setw(5) << std::right << r["features"].get<int>() << " ";
            sbody << std::setw(3) << std::right << r["classes"].get<int>() << " ";
            sbody << std::setw(9) << std::setprecision(2) << std::fixed << r["nodes"].get<float>() << " ";
            sbody << std::setw(9) << std::setprecision(2) << std::fixed << r["leaves"].get<float>() << " ";
            sbody << std::setw(9) << std::setprecision(2) << std::fixed << r["depth"].get<float>() << " ";
            sbody << std::setw(8) << std::right << std::setprecision(6) << std::fixed << r["score"].get<double>() << "±" << std::setw(6) << std::setprecision(4) << std::fixed << r["score_std"].get<double>();
            const std::string status = compareResult(r["dataset"].get<std::string>(), r["score"].get<double>());
            sbody << status;
            sbody << std::setw(12) << std::right << std::setprecision(6) << std::fixed << r["time"].get<double>() << "±" << std::setw(6) << std::setprecision(4) << std::fixed << r["time_std"].get<double>() << " ";
            sbody << r["hyperparameters"].dump();
            sbody << std::endl;
            sbody << std::flush;
            lastResult = r;
            totalScore += r["score"].get<double>();
        }
        if (data["results"].size() == 1 || selectedIndex != -1) {
            sbody << std::string(MAXL, '*') << std::endl;
            if (lastResult.find("notes") != lastResult.end()) {
                if (lastResult["notes"].size() > 0) {
                    sbody << headerLine("Notes: ");
                    for (const auto& note : lastResult["notes"]) {
                        sbody << headerLine(note.get<std::string>());
                    }
                }
            }
            sbody << headerLine(fVector("Train scores: ", lastResult["scores_train"], 14, 12));
            sbody << headerLine(fVector("Test  scores: ", lastResult["scores_test"], 14, 12));
            sbody << headerLine(fVector("Train  times: ", lastResult["times_train"], 10, 3));
            sbody << headerLine(fVector("Test   times: ", lastResult["times_test"], 10, 3));
        } else {
            footer(totalScore);
        }
        sbody << std::string(MAXL, '*') << Colors::RESET() << std::endl;
    }
    void ReportConsole::body()
    {
        std::cout << sbody.str();
    }
    void ReportConsole::showSummary()
    {
        for (const auto& item : summary) {
            std::stringstream oss;
            oss << std::setw(3) << std::left << item.first;
            oss << std::setw(3) << std::right << item.second << " ";
            oss << std::left << meaning.at(item.first);
            sbody << headerLine(oss.str(), 2);
        }
    }

    void ReportConsole::footer(double totalScore)
    {
        sbody << Colors::MAGENTA() << std::string(MAXL, '*') << std::endl;
        showSummary();
        auto score = data["score_name"].get<std::string>();
        auto best = BestScore::getScore(score);
        if (best.first != "") {
            std::stringstream oss;
            oss << score << " compared to " << best.first << " .:  " << totalScore / best.second;
            sbody << headerLine(oss.str());
        }
        if (!getExistBestFile() && compare) {
            std::cout << headerLine("*** Best Results File not found. Couldn't compare any result!");
        }
    }
}