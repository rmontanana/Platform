#include <algorithm>
#include "common/Paths.h"
#include "ResultsManager.h"

namespace platform {
    ResultsManager::ResultsManager(const std::string& model, const std::string& score, bool complete, bool partial) :
        path(Paths::results()), model(model), scoreName(score), complete(complete), partial(partial), maxModel(0), maxTitle(0)
    {
    }
    void ResultsManager::load()
    {
        using std::filesystem::directory_iterator;
        for (const auto& file : directory_iterator(path)) {
            auto filename = file.path().filename().string();
            if (filename.find(".json") != std::string::npos && filename.find("results_") == 0) {
                auto result = Result();
                result.load(path, filename);
                bool addResult = true;
                if (model != "any" && result.getModel() != model || scoreName != "any" && scoreName != result.getScoreName() || complete && !result.isComplete() || partial && result.isComplete())
                    addResult = false;
                if (addResult)
                    files.push_back(result);
            }
        }
        maxModel = std::max(size_t(5), (*max_element(files.begin(), files.end(), [](const Result& a, const Result& b) { return a.getModel().size() < b.getModel().size(); })).getModel().size());
        maxTitle = std::max(size_t(5), (*max_element(files.begin(), files.end(), [](const Result& a, const Result& b) { return a.getTitle().size() < b.getTitle().size(); })).getTitle().size());
    }
    void ResultsManager::hideResult(int index, const std::string& pathHidden)
    {
        auto filename = files.at(index).getFilename();
        rename((path + "/" + filename).c_str(), (pathHidden + "/" + filename).c_str());
        files.erase(files.begin() + index);
    }
    void ResultsManager::deleteResult(int index)
    {
        auto filename = files.at(index).getFilename();
        remove((path + "/" + filename).c_str());
        files.erase(files.begin() + index);
    }
    int ResultsManager::size() const
    {
        return files.size();
    }
    void ResultsManager::sortDate()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            if (a.getDate() == b.getDate()) {
                return a.getModel() < b.getModel();
            }
            return a.getDate() > b.getDate();
            });
    }
    void ResultsManager::sortModel()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            if (a.getModel() == b.getModel()) {
                return a.getDate() > b.getDate();
            }
            return a.getModel() > b.getModel();
            });
    }
    void ResultsManager::sortDuration()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            return a.getDuration() > b.getDuration();
            });
    }
    void ResultsManager::sortScore()
    {
        sort(files.begin(), files.end(), [](const Result& a, const Result& b) {
            if (a.getScore() == b.getScore()) {
                return a.getDate() > b.getDate();
            }
            return a.getScore() > b.getScore();
            });
    }
    bool ResultsManager::empty() const
    {
        return files.empty();
    }
}