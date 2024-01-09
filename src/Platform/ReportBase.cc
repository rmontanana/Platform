#include <sstream>
#include <locale>
#include "Datasets.h"
#include "ReportBase.h"
#include "DotEnv.h"

namespace platform {
    ReportBase::ReportBase(json data_, bool compare) : data(data_), compare(compare), margin(0.1)
    {
        std::stringstream oss;
        oss << "Better than ZeroR + " << std::setprecision(1) << fixed << margin * 100 << "%";
        meaning = {
            {Symbols::equal_best, "Equal to best"},
            {Symbols::better_best, "Better than best"},
            {Symbols::cross, "Less than or equal to ZeroR"},
            {Symbols::upward_arrow, oss.str()}
        };
    }
    std::string ReportBase::fromVector(const std::string& key)
    {
        std::stringstream oss;
        std::string sep = "";
        oss << "[";
        for (auto& item : data[key]) {
            oss << sep << item.get<double>();
            sep = ", ";
        }
        oss << "]";
        return oss.str();
    }
    std::string ReportBase::fVector(const std::string& title, const json& data, const int width, const int precision)
    {
        std::stringstream oss;
        std::string sep = "";
        oss << title << "[";
        for (const auto& item : data) {
            oss << sep << fixed << setw(width) << std::setprecision(precision) << item.get<double>();
            sep = ", ";
        }
        oss << "]";
        return oss.str();
    }
    void ReportBase::show()
    {
        header();
        body();
    }
    std::string ReportBase::compareResult(const std::string& dataset, double result)
    {
        std::string status = " ";
        if (compare) {
            double best = bestResult(dataset, data["model"].get<std::string>());
            if (result == best) {
                status = Symbols::equal_best;
            } else if (result > best) {
                status = Symbols::better_best;
            }
        } else {
            if (data["score_name"].get<std::string>() == "accuracy") {
                auto dt = Datasets(false, Paths::datasets());
                dt.loadDataset(dataset);
                auto numClasses = dt.getNClasses(dataset);
                if (numClasses == 2) {
                    std::vector<int> distribution = dt.getClassesCounts(dataset);
                    double nSamples = dt.getNSamples(dataset);
                    std::vector<int>::iterator maxValue = max_element(distribution.begin(), distribution.end());
                    double mark = *maxValue / nSamples * (1 + margin);
                    if (mark > 1) {
                        mark = 0.9995;
                    }
                    status = result < mark ? Symbols::cross : result > mark ? Symbols::upward_arrow : "=";
                }
            }
        }
        if (status != " ") {
            auto item = summary.find(status);
            if (item != summary.end()) {
                summary[status]++;
            } else {
                summary[status] = 1;
            }
        }
        return status;
    }
    double ReportBase::bestResult(const std::string& dataset, const std::string& model)
    {
        double value = 0.0;
        if (bestResults.size() == 0) {
            // try to load the best results
            std::string score = data["score_name"];
            replace(score.begin(), score.end(), '_', '-');
            std::string fileName = "best_results_" + score + "_" + model + ".json";
            ifstream resultData(Paths::results() + "/" + fileName);
            if (resultData.is_open()) {
                bestResults = json::parse(resultData);
            } else {
                existBestFile = false;
            }
        }
        try {
            value = bestResults.at(dataset).at(0);
        }
        catch (exception) {
            value = 1.0;

        }
        return value;
    }
    bool ReportBase::getExistBestFile()
    {
        return existBestFile;
    }
}