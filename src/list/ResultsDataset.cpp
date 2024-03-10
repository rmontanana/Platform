#include <algorithm>
#include "common/Paths.h"
#include "ResultsDataset.h"

namespace platform {
    ResultsDataset::ResultsDataset(const std::string& dataset, const std::string& model, const std::string& score) :
        path(Paths::results()), dataset(dataset), model(model), scoreName(score), maxModel(0), maxFile(0), maxHyper(15), maxResult(0)
    {
    }
    void ResultsDataset::load()
    {
        using std::filesystem::directory_iterator;
        for (const auto& file : directory_iterator(path)) {
            auto filename = file.path().filename().string();
            if (filename.find(".json") != std::string::npos && filename.find("results_") == 0) {
                auto result = Result();
                result.load(path, filename);
                if (model != "any" && result.getModel() != model)
                    continue;
                auto data = result.getData()["results"];
                for (auto const& item : data) {
                    if (item["dataset"] == dataset) {
                        auto hyper_length = item["hyperparameters"].dump().size();
                        if (hyper_length > maxHyper)
                            maxHyper = hyper_length;
                        if (item["score"].get<double>() > maxResult)
                            maxResult = item["score"].get<double>();
                        files.push_back(result);
                        break;
                    }
                }
            }
        }
        maxModel = std::max(size_t(5), (*max_element(files.begin(), files.end(), [](const Result& a, const Result& b) { return a.getModel().size() < b.getModel().size(); })).getModel().size());
        maxFile = std::max(size_t(4), (*max_element(files.begin(), files.end(), [](const Result& a, const Result& b) { return a.getFilename().size() < b.getFilename().size(); })).getFilename().size());
    }
    int ResultsDataset::size() const
    {
        return files.size();
    }
    void ResultsDataset::sortModel()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            if (a.getModel() == b.getModel()) {
                return a.getDate() > b.getDate();
            }
            return a.getModel() < b.getModel();
            });
    }
    bool ResultsDataset::empty() const
    {
        return files.empty();
    }
}