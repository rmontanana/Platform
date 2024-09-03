#include <filesystem>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "common/Colors.h"
#include "common/CLocale.h"
#include "common/Paths.h"
#include "common/Utils.h" // compute_std
#include "results/Result.h"
#include "BestResultsExcel.h"
#include "BestResultsTex.h"
#include "best/Statistics.h"
#include "BestResults.h"


namespace fs = std::filesystem;
// function ftime_to_std::string, Code taken from 
// https://stackoverflow.com/a/58237530/1389271
template <typename TP>
std::string ftime_to_string(TP tp)
{
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(tp - TP::clock::now()
        + std::chrono::system_clock::now());
    auto tt = std::chrono::system_clock::to_time_t(sctp);
    std::tm* gmt = std::gmtime(&tt);
    std::stringstream buffer;
    buffer << std::put_time(gmt, "%Y-%m-%d %H:%M");
    return buffer.str();
}
namespace platform {
    std::string BestResults::build()
    {
        auto files = loadResultFiles();
        if (files.size() == 0) {
            std::cerr << Colors::MAGENTA() << "No result files were found!" << Colors::RESET() << std::endl;
            exit(1);
        }
        json bests;
        for (const auto& file : files) {
            auto result = Result();
            result.load(path, file);
            auto data = result.getJson();
            for (auto const& item : data.at("results")) {
                bool update = true;
                auto datasetName = item.at("dataset").get<std::string>();
                if (dataset != "any" && dataset != datasetName) {
                    continue;
                }
                if (bests.contains(datasetName)) {
                    if (item.at("score").get<double>() < bests[datasetName].at(0).get<double>()) {
                        update = false;
                    }
                }
                if (update) {
                    bests[datasetName] = { item.at("score").get<double>(), item.at("hyperparameters"), file, item.at("score_std").get<double>() };
                }
            }
        }
        if (bests.empty()) {
            std::cerr << Colors::MAGENTA() << "No results found for model " << model << " and score " << score << Colors::RESET() << std::endl;
            exit(1);
        }
        std::string bestFileName = path + Paths::bestResultsFile(score, model);
        std::ofstream file(bestFileName);
        file << bests;
        file.close();
        return bestFileName;
    }
    std::pair<std::string, std::string> getModelScore(std::string name)
    {
        // results_accuracy_BoostAODE_MacBookpro16_2023-09-06_12:27:00_1.json
        auto pos = name.find("_");
        auto pos2 = name.find("_", pos + 1);
        std::string score = name.substr(pos + 1, pos2 - pos - 1);
        pos = name.find("_", pos2 + 1);
        std::string model = name.substr(pos2 + 1, pos - pos2 - 1);
        return { model, score };
    }
    std::vector<std::string> BestResults::loadResultFiles()
    {
        std::vector<std::string> files;
        using std::filesystem::directory_iterator;
        std::string fileModel, fileScore;
        for (const auto& file : directory_iterator(path)) {
            auto fileName = file.path().filename().string();
            if (fileName.find(".json") != std::string::npos && fileName.find("results_") == 0) {
                tie(fileModel, fileScore) = getModelScore(fileName);
                if (score == fileScore && (model == fileModel || model == "any")) {
                    files.push_back(fileName);
                }
            }
        }
        std::sort(files.begin(), files.end());
        return files;
    }
    json BestResults::loadFile(const std::string& fileName)
    {
        std::ifstream resultData(fileName);
        if (resultData.is_open()) {
            json data = json::parse(resultData);
            return data;
        }
        throw std::invalid_argument("Unable to open result file. [" + fileName + "]");
    }
    std::vector<std::string> BestResults::getModels()
    {
        std::set<std::string> models;
        std::vector<std::string> result;
        auto files = loadResultFiles();
        if (files.size() == 0) {
            std::cerr << Colors::MAGENTA() << "No result files were found!" << Colors::RESET() << std::endl;
            exit(1);
        }
        std::string fileModel, fileScore;
        for (const auto& file : files) {
            // extract the model from the file name
            tie(fileModel, fileScore) = getModelScore(file);
            // add the model to the std::vector of models
            models.insert(fileModel);
        }
        result = std::vector<std::string>(models.begin(), models.end());
        maxModelName = (*max_element(result.begin(), result.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size();
        maxModelName = std::max(12, maxModelName);
        return result;
    }
    std::vector<std::string> BestResults::getDatasets(json table)
    {
        std::vector<std::string> datasets;
        for (const auto& dataset_ : table.items()) {
            datasets.push_back(dataset_.key());
        }
        maxDatasetName = (*max_element(datasets.begin(), datasets.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size();
        maxDatasetName = std::max(7, maxDatasetName);
        return datasets;
    }
    void BestResults::buildAll()
    {
        auto models = getModels();
        std::cout << "Building best results for model: ";
        for (const auto& model : models) {
            this->model = model;
            std::cout << model << ", ";
            build();
        }
        std::cout << "end." << std::endl << std::endl;
        model = "any";
    }
    void BestResults::listFile()
    {
        std::string bestFileName = path + Paths::bestResultsFile(score, model);
        if (FILE* fileTest = fopen(bestFileName.c_str(), "r")) {
            fclose(fileTest);
        } else {
            std::cerr << Colors::MAGENTA() << "File " << bestFileName << " doesn't exist." << Colors::RESET() << std::endl;
            exit(1);
        }
        auto temp = ConfigLocale();
        auto date = ftime_to_string(std::filesystem::last_write_time(bestFileName));
        auto data = loadFile(bestFileName);
        auto datasets = getDatasets(data);
        int maxFileName = 0;
        int maxHyper = 15;
        for (auto const& item : data.items()) {
            maxHyper = std::max(maxHyper, (int)item.value().at(1).dump().size());
            maxFileName = std::max(maxFileName, (int)item.value().at(2).get<std::string>().size());
        }
        std::stringstream oss;
        oss << Colors::GREEN() << "Best results for " << model << " as of " << date << std::endl;
        std::cout << oss.str();
        std::cout << std::string(oss.str().size() - 8, '-') << std::endl;
        std::cout << Colors::GREEN() << " #  " << std::setw(maxDatasetName + 1) << std::left << "Dataset" << "Score       " << std::setw(maxFileName) << "File" << " Hyperparameters" << std::endl;
        std::cout << "=== " << std::string(maxDatasetName, '=') << " =========== " << std::string(maxFileName, '=') << " " << std::string(maxHyper, '=') << std::endl;
        auto i = 0;
        double total = 0;
        for (auto const& item : data.items()) {
            auto color = (i % 2) ? Colors::BLUE() : Colors::CYAN();
            double value = item.value().at(0).get<double>();
            std::cout << color << std::setw(3) << std::fixed << std::right << i++ << " ";
            std::cout << std::setw(maxDatasetName) << std::left << item.key() << " ";
            std::cout << std::setw(11) << std::setprecision(9) << std::fixed << value << " ";
            std::cout << std::setw(maxFileName) << item.value().at(2).get<std::string>() << " ";
            std::cout << item.value().at(1) << " ";
            std::cout << std::endl;
            total += value;
        }
        std::cout << Colors::GREEN() << "=== " << std::string(maxDatasetName, '=') << " ===========" << std::endl;
        std::cout << Colors::GREEN() << "    Total" << std::string(maxDatasetName - 5, '.') << " " << std::setw(11) << std::setprecision(8) << std::fixed << total << std::endl;

    }
    json BestResults::buildTableResults(std::vector<std::string> models)
    {
        json table;
        auto maxDate = std::filesystem::file_time_type::max();
        for (const auto& model : models) {
            this->model = model;
            std::string bestFileName = path + Paths::bestResultsFile(score, model);
            if (FILE* fileTest = fopen(bestFileName.c_str(), "r")) {
                fclose(fileTest);
            } else {
                std::cerr << Colors::MAGENTA() << "File " << bestFileName << " doesn't exist." << Colors::RESET() << std::endl;
                exit(1);
            }
            auto dateWrite = std::filesystem::last_write_time(bestFileName);
            if (dateWrite < maxDate) {
                maxDate = dateWrite;
            }
            auto data = loadFile(bestFileName);
            table[model] = data;
        }
        table["dateTable"] = ftime_to_string(maxDate);
        return table;
    }

    void BestResults::printTableResults(std::vector<std::string> models, json table, bool tex)
    {
        std::stringstream oss;
        oss << Colors::GREEN() << "Best results for " << score << " as of " << table.at("dateTable").get<std::string>() << std::endl;
        std::cout << oss.str();
        std::cout << std::string(oss.str().size() - 8, '-') << std::endl;
        std::cout << Colors::GREEN() << " #  " << std::setw(maxDatasetName + 1) << std::left << std::string("Dataset");
        auto bestResultsTex = BestResultsTex();
        if (tex) {
            bestResultsTex.results_header(models, table.at("dateTable").get<std::string>());
        }
        for (const auto& model : models) {
            std::cout << std::setw(maxModelName) << std::left << model << " ";
        }
        std::cout << std::endl;
        std::cout << "=== " << std::string(maxDatasetName, '=') << " ";
        for (const auto& model : models) {
            std::cout << std::string(maxModelName, '=') << " ";
        }
        std::cout << std::endl;
        auto i = 0;
        std::map<std::string, std::vector<double>> totals;
        int nDatasets = table.begin().value().size();
        auto datasets = getDatasets(table.begin().value());
        if (tex) {
            bestResultsTex.results_body(datasets, table);
        }
        for (auto const& dataset_ : datasets) {
            auto color = (i % 2) ? Colors::BLUE() : Colors::CYAN();
            std::cout << color << std::setw(3) << std::fixed << std::right << i++ << " ";
            std::cout << std::setw(maxDatasetName) << std::left << dataset_ << " ";
            double maxValue = 0;
            // Find out the max value for this dataset
            for (const auto& model : models) {
                double value;
                try {
                    value = table[model].at(dataset_).at(0).get<double>();
                }
                catch (nlohmann::json_abi_v3_11_3::detail::out_of_range err) {
                    value = -1.0;
                }
                if (value > maxValue) {
                    maxValue = value;
                }
            }
            // Print the row with red colors on max values
            for (const auto& model : models) {
                std::string efectiveColor = color;
                double value;
                try {
                    value = table[model].at(dataset_).at(0).get<double>();
                }
                catch (nlohmann::json_abi_v3_11_3::detail::out_of_range err) {
                    value = -1.0;
                }
                if (value == maxValue) {
                    efectiveColor = Colors::RED();
                }
                if (value == -1) {
                    std::cout << Colors::YELLOW() << std::setw(maxModelName) << std::right << "N/A" << " ";
                } else {
                    totals[model].push_back(value);
                    std::cout << efectiveColor << std::setw(maxModelName) << std::setprecision(maxModelName - 2) << std::fixed << value << " ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << Colors::GREEN() << "=== " << std::string(maxDatasetName, '=') << " ";
        for (const auto& model : models) {
            std::cout << std::string(maxModelName, '=') << " ";
        }
        std::cout << std::endl;
        std::cout << Colors::GREEN() << "    Average" << std::string(maxDatasetName - 7, '.') << " ";
        double max_value = 0.0;
        std::string best_model = "";
        for (const auto& total : totals) {
            auto actual = std::reduce(total.second.begin(), total.second.end());
            if (actual > max_value) {
                max_value = actual;
                best_model = total.first;
            }
        }
        if (tex) {
            bestResultsTex.results_footer(totals, best_model);
        }
        for (const auto& model : models) {
            std::string efectiveColor = model == best_model ? Colors::RED() : Colors::GREEN();
            double value = std::reduce(totals[model].begin(), totals[model].end()) / nDatasets;
            double std_value = compute_std(totals[model], value);
            std::cout << efectiveColor << std::right << std::setw(maxModelName) << std::setprecision(maxModelName - 4) << std::fixed << value << " ";

        }
        std::cout << std::endl;
    }
    void BestResults::reportSingle(bool excel)
    {
        listFile();
        if (excel) {
            auto models = getModels();
            // Build the table of results
            json table = buildTableResults(models);
            std::vector<std::string> datasets = getDatasets(table.begin().value());
            BestResultsExcel excel_report(score, datasets);
            excel_report.reportSingle(model, path + Paths::bestResultsFile(score, model));
            messageOutputFile("Excel", excel_report.getFileName());
        }
    }
    void BestResults::reportAll(bool excel, bool tex)
    {
        auto models = getModels();
        // Build the table of results
        json table = buildTableResults(models);
        std::vector<std::string> datasets = getDatasets(table.begin().value());
        // Print the table of results
        printTableResults(models, table, tex);
        // Compute the Friedman test
        std::map<std::string, std::map<std::string, float>> ranksModels;
        if (friedman) {
            Statistics stats(models, datasets, table, significance);
            auto result = stats.friedmanTest();
            stats.postHocHolmTest(result, tex);
            ranksModels = stats.getRanks();
        }
        if (tex) {
            messageOutputFile("TeX", Paths::tex() + Paths::tex_output());
            if (friedman) {
                messageOutputFile("TeX", Paths::tex() + Paths::tex_post_hoc());
            }
        }
        if (excel) {
            BestResultsExcel excel(score, datasets);
            excel.reportAll(models, table, ranksModels, friedman, significance);
            if (friedman) {
                int idx = -1;
                double min = 2000;
                // Find out the control model
                auto totals = std::vector<double>(models.size(), 0.0);
                for (const auto& dataset_ : datasets) {
                    for (int i = 0; i < models.size(); ++i) {
                        totals[i] += ranksModels[dataset_][models[i]];
                    }
                }
                for (int i = 0; i < models.size(); ++i) {
                    if (totals[i] < min) {
                        min = totals[i];
                        idx = i;
                    }
                }
                model = models.at(idx);
                excel.reportSingle(model, path + Paths::bestResultsFile(score, model));
            }
            messageOutputFile("Excel", excel.getFileName());
        }
    }
    void BestResults::messageOutputFile(const std::string& title, const std::string& fileName)
    {
        std::cout << Colors::YELLOW() << "** " << std::setw(5) << std::left << title
            << " file generated: " << fileName << Colors::RESET() << std::endl;
    }
}