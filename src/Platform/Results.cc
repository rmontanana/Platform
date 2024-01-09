#include "Results.h"
#include <algorithm>

namespace platform {
    Results::Results(const std::string& path, const std::string& model, const std::string& score, bool complete, bool partial) :
        path(path), model(model), scoreName(score), complete(complete), partial(partial)
    {
        load();
        if (!files.empty()) {
            maxModel = (*max_element(files.begin(), files.end(), [](const Result& a, const Result& b) { return a.getModel().size() < b.getModel().size(); })).getModel().size();
        } else {
            maxModel = 0;
        }
    };
    void Results::load()
    {
        using std::filesystem::directory_iterator;
        for (const auto& file : directory_iterator(path)) {
            auto filename = file.path().filename().string();
            if (filename.find(".json") != std::string::npos && filename.find("results_") == 0) {
                auto result = Result(path, filename);
                bool addResult = true;
                if (model != "any" && result.getModel() != model || scoreName != "any" && scoreName != result.getScoreName() || complete && !result.isComplete() || partial && result.isComplete())
                    addResult = false;
                if (addResult)
                    files.push_back(result);
            }
        }
    }
    void Results::hideResult(int index, const std::string& pathHidden)
    {
        auto filename = files.at(index).getFilename();
        rename((path + "/" + filename).c_str(), (pathHidden + "/" + filename).c_str());
        files.erase(files.begin() + index);
    }
    void Results::deleteResult(int index)
    {
        auto filename = files.at(index).getFilename();
        remove((path + "/" + filename).c_str());
        files.erase(files.begin() + index);
    }
    int Results::size() const
    {
        return files.size();
    }
    void Results::sortDate()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getDate() > b.getDate();
            });
    }
    void Results::sortModel()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getModel() > b.getModel();
            });
    }
    void Results::sortDuration()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getDuration() > b.getDuration();
            });
    }
    void Results::sortScore()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getScore() > b.getScore();
            });
    }
    bool Results::empty() const
    {
        return files.empty();
    }
}