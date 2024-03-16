
#include "common/Colors.h"
#include "results/ResultsDataset.h"
#include "ResultsDatasetConsole.h"
namespace platform {
    void ResultsDatasetsConsole::list_results(const std::string& dataset, const std::string& score, const std::string& model)
    {
        output.str("");
        auto results = platform::ResultsDataset(dataset, model, score);
        results.load();
        results.sortModel();
        if (results.empty()) {
            output << Colors::RED() << "No results found for dataset " << dataset << " and model " << model << Colors::RESET() << std::endl;
            return;
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
        output << Colors::GREEN() << "Results of dataset " << dataset << " - for " << model << " model" << std::endl;
        output << "There are " << results.size() << " results" << std::endl;
        output << Colors::GREEN() << " #  " << std::setw(maxModel + 1) << std::left << "Model" << "Date       Time     Score       Hyperparameters" << std::endl;
        output << "=== " << std::string(maxModel, '=') << " ========== ======== =========== " << std::string(maxHyper, '=') << std::endl;
        numLines = 4;
        auto i = 0;
        for (const auto& item : data["results"]) {
            auto color = (i % 2) ? Colors::BLUE() : Colors::CYAN();
            auto score = item["score"].get<double>();
            color = score == data["max_models"][item["model"].get<std::string>()] ? Colors::YELLOW() : color;
            color = score == maxResult ? Colors::RED() : color;
            output << color << std::setw(3) << std::fixed << std::right << i++ << " ";
            output << std::setw(maxModel) << std::left << item["model"].get<std::string>() << " ";
            output << color << item["date"].get<std::string>() << " ";
            output << color << item["time"].get<std::string>() << " ";
            output << std::setw(11) << std::setprecision(9) << std::fixed << score << " ";
            output << item["hyperparameters"].get<std::string>() << std::endl;
            numLines++;
        }
        data["dataset"] = dataset;
        data["score"] = score;
        data["model"] = model;
        data["lengths"]["maxModel"] = maxModel;
        data["lengths"]["maxHyper"] = maxHyper;
        data["maxResult"] = maxResult;
    }
}




