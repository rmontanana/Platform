#include <filesystem>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "BestResults.h"
#include "Result.h"
#include "Colors.h"
#include "Statistics.h"
#include "BestResultsExcel.h"
#include "CLocale.h"


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
            auto result = Result(path, file);
            auto data = result.load();
            for (auto const& item : data.at("results")) {
                bool update = false;
                // Check if results file contains only one dataset
                auto datasetName = item.at("dataset").get<std::string>();
                if (bests.contains(datasetName)) {
                    if (item.at("score").get<double>() > bests[datasetName].at(0).get<double>()) {
                        update = true;
                    }
                } else {
                    update = true;
                }
                if (update) {
                    bests[datasetName] = { item.at("score").get<double>(), item.at("hyperparameters"), file };
                }
            }
        }
        std::string bestFileName = path + bestResultFile();
        if (FILE* fileTest = fopen(bestFileName.c_str(), "r")) {
            fclose(fileTest);
            std::cout << Colors::MAGENTA() << "File " << bestFileName << " already exists and it shall be overwritten." << Colors::RESET() << std::endl;
        }
        std::ofstream file(bestFileName);
        file << bests;
        file.close();
        return bestFileName;
    }
    std::string BestResults::bestResultFile()
    {
        return "best_results_" + score + "_" + model + ".json";
    }
    std::pair<std::string, std::string> getModelScore(std::string name)
    {
        // results_accuracy_BoostAODE_MacBookpro16_2023-09-06_12:27:00_1.json
        int i = 0;
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
        return result;
    }
    std::vector<std::string> BestResults::getDatasets(json table)
    {
        std::vector<std::string> datasets;
        for (const auto& dataset : table.items()) {
            datasets.push_back(dataset.key());
        }
        return datasets;
    }
    void BestResults::buildAll()
    {
        auto models = getModels();
        for (const auto& model : models) {
            std::cout << "Building best results for model: " << model << std::endl;
            this->model = model;
            build();
        }
        model = "any";
    }
    void BestResults::listFile()
    {
        std::string bestFileName = path + bestResultFile();
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
        int maxDatasetName = (*max_element(datasets.begin(), datasets.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size();
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
        bool odd = true;
        double total = 0;
        for (auto const& item : data.items()) {
            auto color = odd ? Colors::BLUE() : Colors::CYAN();
            double value = item.value().at(0).get<double>();
            std::cout << color << std::setw(3) << std::fixed << std::right << i++ << " ";
            std::cout << std::setw(maxDatasetName) << std::left << item.key() << " ";
            std::cout << std::setw(11) << std::setprecision(9) << std::fixed << value << " ";
            std::cout << std::setw(maxFileName) << item.value().at(2).get<std::string>() << " ";
            std::cout << item.value().at(1) << " ";
            std::cout << std::endl;
            total += value;
            odd = !odd;
        }
        std::cout << Colors::GREEN() << "=== " << std::string(maxDatasetName, '=') << " ===========" << std::endl;
        std::cout << std::setw(5 + maxDatasetName) << "Total.................. " << std::setw(11) << std::setprecision(8) << std::fixed << total << std::endl;
    }
    json BestResults::buildTableResults(std::vector<std::string> models)
    {
        json table;
        auto maxDate = std::filesystem::file_time_type::max();
        for (const auto& model : models) {
            this->model = model;
            std::string bestFileName = path + bestResultFile();
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
    void BestResults::printTableResults(std::vector<std::string> models, json table)
    {
        std::stringstream oss;
        oss << Colors::GREEN() << "Best results for " << score << " as of " << table.at("dateTable").get<std::string>() << std::endl;
        std::cout << oss.str();
        std::cout << std::string(oss.str().size() - 8, '-') << std::endl;
        std::cout << Colors::GREEN() << " #  " << std::setw(maxDatasetName + 1) << std::left << std::string("Dataset");
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
        bool odd = true;
        std::map<std::string, double> totals;
        int nDatasets = table.begin().value().size();
        for (const auto& model : models) {
            totals[model] = 0.0;
        }
        auto datasets = getDatasets(table.begin().value());
        for (auto const& dataset : datasets) {
            auto color = odd ? Colors::BLUE() : Colors::CYAN();
            std::cout << color << std::setw(3) << std::fixed << std::right << i++ << " ";
            std::cout << std::setw(maxDatasetName) << std::left << dataset << " ";
            double maxValue = 0;
            // Find out the max value for this dataset
            for (const auto& model : models) {
                double value = table[model].at(dataset).at(0).get<double>();
                if (value > maxValue) {
                    maxValue = value;
                }
            }
            // Print the row with red colors on max values
            for (const auto& model : models) {
                std::string efectiveColor = color;
                double value = table[model].at(dataset).at(0).get<double>();
                if (value == maxValue) {
                    efectiveColor = Colors::RED();
                }
                totals[model] += value;
                std::cout << efectiveColor << std::setw(maxModelName) << std::setprecision(maxModelName - 2) << std::fixed << value << " ";
            }
            std::cout << std::endl;
            odd = !odd;
        }
        std::cout << Colors::GREEN() << "=== " << std::string(maxDatasetName, '=') << " ";
        for (const auto& model : models) {
            std::cout << std::string(maxModelName, '=') << " ";
        }
        std::cout << std::endl;
        std::cout << Colors::GREEN() << std::setw(5 + maxDatasetName) << "    Totals...................";
        double max = 0.0;
        for (const auto& total : totals) {
            if (total.second > max) {
                max = total.second;
            }
        }
        for (const auto& model : models) {
            std::string efectiveColor = Colors::GREEN();
            if (totals[model] == max) {
                efectiveColor = Colors::RED();
            }
            std::cout << efectiveColor << std::right << std::setw(maxModelName) << std::setprecision(maxModelName - 4) << std::fixed << totals[model] << " ";
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
            BestResultsExcel excel(score, datasets);
            excel.reportSingle(model, path + bestResultFile());
            messageExcelFile(excel.getFileName());
        }
    }
    void BestResults::reportAll(bool excel)
    {
        auto models = getModels();
        // Build the table of results
        json table = buildTableResults(models);
        std::vector<std::string> datasets = getDatasets(table.begin().value());
        maxModelName = (*max_element(models.begin(), models.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size();
        maxModelName = std::max(12, maxModelName);
        maxDatasetName = (*max_element(datasets.begin(), datasets.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size();
        maxDatasetName = std::max(25, maxDatasetName);
        // Print the table of results
        printTableResults(models, table);
        // Compute the Friedman test
        std::map<std::string, std::map<std::string, float>> ranksModels;
        if (friedman) {
            Statistics stats(models, datasets, table, significance);
            auto result = stats.friedmanTest();
            stats.postHocHolmTest(result);
            ranksModels = stats.getRanks();
        }
        if (excel) {
            BestResultsExcel excel(score, datasets);
            excel.reportAll(models, table, ranksModels, friedman, significance);
            if (friedman) {
                int idx = -1;
                double min = 2000;
                // Find out the control model
                auto totals = std::vector<double>(models.size(), 0.0);
                for (const auto& dataset : datasets) {
                    for (int i = 0; i < models.size(); ++i) {
                        totals[i] += ranksModels[dataset][models[i]];
                    }
                }
                for (int i = 0; i < models.size(); ++i) {
                    if (totals[i] < min) {
                        min = totals[i];
                        idx = i;
                    }
                }
                model = models.at(idx);
                excel.reportSingle(model, path + bestResultFile());
            }
            messageExcelFile(excel.getFileName());
        }
    }
    void BestResults::messageExcelFile(const std::string& fileName)
    {
        std::cout << Colors::YELLOW() << "** Excel file generated: " << fileName << Colors::RESET() << std::endl;
    }
}