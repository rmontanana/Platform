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
        do_header();
    }
    void ReportConsole::body()
    {
        do_body();
        std::cout << sbody.str();
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
        vbody.clear();
        auto tmp = ConfigLocale();
        int maxHyper = 15;
        int maxDataset = 7;
        for (const auto& r : data["results"]) {
            maxHyper = std::max(maxHyper, (int)r["hyperparameters"].dump().size());
            maxDataset = std::max(maxDataset, (int)r["dataset"].get<std::string>().size());
        }
        std::vector<std::string> header_labels = { " #", "Dataset", "Sampl.", "Feat.", "Cls", nodes_label, leaves_label, depth_label, "Score", "Time", "Hyperparameters" };
        sheader << Colors::GREEN();
        std::vector<int> header_lengths = { 3, maxDataset, 6, 5, 3, 9, 9, 9, 15, 20, maxHyper };
        for (int i = 0; i < header_labels.size(); i++) {
            sheader << std::setw(header_lengths[i]) << std::left << header_labels[i] << " ";
        }
        sheader << std::endl;
        for (int i = 0; i < header_labels.size(); i++) {
            sheader << std::string(header_lengths[i], '=') << " ";
        }
        sheader << std::endl;
        std::cout << sheader.str();
        json lastResult;
        double totalScore = 0.0;
        int index = 0;
        for (const auto& r : data["results"]) {
            if (selectedIndex != -1 && index != selectedIndex) {
                index++;
                continue;
            }
            auto color = (index % 2) ? Colors::CYAN() : Colors::BLUE();
            std::stringstream line;
            line << color;
            line << std::setw(3) << std::right << index++ << " ";
            line << std::setw(maxDataset) << std::left << r["dataset"].get<std::string>() << " ";
            line << std::setw(6) << std::right << r["samples"].get<int>() << " ";
            line << std::setw(5) << std::right << r["features"].get<int>() << " ";
            line << std::setw(3) << std::right << r["classes"].get<int>() << " ";
            line << std::setw(9) << std::setprecision(2) << std::fixed << r["nodes"].get<float>() << " ";
            line << std::setw(9) << std::setprecision(2) << std::fixed << r["leaves"].get<float>() << " ";
            line << std::setw(9) << std::setprecision(2) << std::fixed << r["depth"].get<float>() << " ";
            line << std::setw(8) << std::right << std::setprecision(6) << std::fixed << r["score"].get<double>() << "±" << std::setw(6) << std::setprecision(4) << std::fixed << r["score_std"].get<double>();
            const std::string status = compareResult(r["dataset"].get<std::string>(), r["score"].get<double>());
            line << status;
            line << std::setw(12) << std::right << std::setprecision(6) << std::fixed << r["time"].get<double>() << "±" << std::setw(7) << std::setprecision(4) << std::fixed << r["time_std"].get<double>() << " ";
            line << r["hyperparameters"].dump();
            line << std::endl;
            vbody.push_back(line.str());
            sbody << line.str();
            lastResult = r;
            totalScore += r["score"].get<double>();
        }
        if (data["results"].size() == 1 || selectedIndex != -1) {
            std::stringstream line;
            line << Colors::MAGENTA() << std::string(MAXL, '*') << std::endl;
            vbody.push_back(line.str());
            sbody << line.str();
            if (lastResult.find("notes") != lastResult.end()) {
                if (lastResult["notes"].size() > 0) {
                    sbody << headerLine("Notes: ");
                    vbody.push_back(headerLine("Notes: "));
                    for (const auto& note : lastResult["notes"]) {
                        line.str("");
                        line << headerLine(note.get<std::string>());
                        vbody.push_back(line.str());
                        sbody << line.str();
                    }
                }
            }
            line.str(""); line << headerLine(fVector("Train scores: ", lastResult["scores_train"], 14, 12));
            vbody.push_back(line.str()); sbody << line.str();
            line.str(""); line << headerLine(fVector("Test  scores: ", lastResult["scores_test"], 14, 12));
            vbody.push_back(line.str()); sbody << line.str();
            line.str(""); line << headerLine(fVector("Train  times: ", lastResult["times_train"], 10, 3));
            vbody.push_back(line.str()); sbody << line.str();
            line.str(""); line << headerLine(fVector("Test   times: ", lastResult["times_test"], 10, 3));
            vbody.push_back(line.str()); sbody << line.str();

        } else {
            footer(totalScore);
        }
        sbody << std::string(MAXL, '*') << Colors::RESET() << std::endl;
        vbody.push_back(std::string(MAXL, '*') + Colors::RESET() + "\n");
        if (data["results"].size() == 1 || selectedIndex != -1) {
            vbody.push_back(showClassificationReport(Colors::BLUE()));
        }
    }
    void ReportConsole::showSummary()
    {
        for (const auto& item : summary) {
            std::stringstream oss;
            oss << std::setw(3) << std::left << item.first;
            oss << std::setw(3) << std::right << item.second << " ";
            oss << std::left << meaning.at(item.first);
            sbody << headerLine(oss.str(), 2);
            vbody.push_back(headerLine(oss.str(), 2));
        }
    }

    void ReportConsole::footer(double totalScore)
    {
        std::stringstream linea;
        linea << Colors::MAGENTA() << std::string(MAXL, '*') << std::endl;
        vbody.push_back(linea.str()); sbody << linea.str();
        showSummary();
        auto score = data["score_name"].get<std::string>();
        auto best = BestScore::getScore(score);
        if (best.first != "") {
            std::stringstream oss;
            oss << score << " compared to " << best.first << " .:  " << totalScore / best.second;
            sbody << headerLine(oss.str());
            vbody.push_back(headerLine(oss.str()));
        }
        if (!getExistBestFile() && compare) {
            std::cout << headerLine("*** Best Results File not found. Couldn't compare any result!");
        }
    }
    Scores ReportConsole::aggregateScore(std::string key)
    {
        auto lastResult = data["results"][0];
        auto item = data["results"][0];
        auto scores = Scores(item[key][0]);
        for (int i = 1; i < item[key].size(); i++) {
            auto score = Scores(item[key][i]);
            scores.aggregate(score);
        }
        return scores;
    }
    std::string ReportConsole::showClassificationReport(std::string color)
    {
        std::stringstream oss;
        auto result = data["results"][0];
        if (result.find("confusion_matrices") == result.end())
            return "";
        bool second_header = false;
        int lines_header = 0;
        std::string color_line;
        std::string suffix = "";
        auto scores = aggregateScore("confusion_matrices");
        auto output_test = scores.classification_report(color, "Test");
        int maxLine = (*std::max_element(output_test.begin(), output_test.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size();
        bool train_data = result.find("confusion_matrices_train") != result.end();
        std::vector<std::string> output_train;
        if (train_data) {
            auto scores_train = aggregateScore("confusion_matrices_train");
            output_train = scores_train.classification_report(color, "Train");
        }
        oss << Colors::BLUE();
        for (int i = 0; i < output_test.size(); i++) {
            if (i < 2 || second_header) {
                color_line = Colors::GREEN();
            } else {
                color_line = Colors::BLUE();
                if (lines_header > 1)
                    suffix = std::string(14, ' '); // compensate for the color
            }
            if (train_data) {
                oss << color_line << std::left << std::setw(maxLine) << output_train[i]
                    << suffix << Colors::BLUE() << " | " << color_line << std::left << std::setw(maxLine)
                    << output_test[i] << std::endl;
            } else {
                oss << color_line << output_test[i] << std::endl;
            }
            if (output_test[i] == "" || (second_header && lines_header < 2)) {
                lines_header++;
                second_header = true;
            } else {
                second_header = false;
            }
        }
        oss << Colors::RESET();
        return oss.str();
    }
}