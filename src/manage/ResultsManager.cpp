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
        bool found = false;
        for (const auto& file : directory_iterator(path)) {
            auto filename = file.path().filename().string();
            if (filename.find(".json") != std::string::npos && filename.find("results_") == 0) {
                auto result = Result();
                result.load(path, filename);
                bool addResult = true;
                if (model != "any" && result.getModel() != model || scoreName != "any" && scoreName != result.getScoreName() || complete && !result.isComplete() || partial && result.isComplete())
                    addResult = false;
                if (addResult) {
                    files.push_back(result);
                    found = true;
                }
            }
        }
        if (found) {
            maxModel = std::max(size_t(5), (*max_element(files.begin(), files.end(), [](const Result& a, const Result& b) { return a.getModel().size() < b.getModel().size(); })).getModel().size());
            maxTitle = std::max(size_t(5), (*max_element(files.begin(), files.end(), [](const Result& a, const Result& b) { return a.getTitle().size() < b.getTitle().size(); })).getTitle().size());
        }
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
    void ResultsManager::sortDate(SortType type)
    {
        if (empty())
            return;
        sort(files.begin(), files.end(), [type](const Result& a, const Result& b) {
            if (a.getDate() == b.getDate()) {
                if (type == SortType::ASC)
                    return a.getModel() < b.getModel();
                return a.getModel() > b.getModel();
            }
            if (type == SortType::ASC)
                return a.getDate() < b.getDate();
            return a.getDate() > b.getDate();
            });
    }
    void ResultsManager::sortModel(SortType type)
    {
        if (empty())
            return;
        sort(files.begin(), files.end(), [type](const Result& a, const Result& b) {
            if (a.getModel() == b.getModel()) {
                if (type == SortType::ASC)
                    return a.getDate() < b.getDate();
                return a.getDate() > b.getDate();
            }
            if (type == SortType::ASC)
                return a.getModel() < b.getModel();
            return a.getModel() > b.getModel();
            });
    }
    void ResultsManager::sortDuration(SortType type)
    {
        if (empty())
            return;
        sort(files.begin(), files.end(), [type](const Result& a, const Result& b) {
            if (type == SortType::ASC)
                return a.getDuration() < b.getDuration();
            return a.getDuration() > b.getDuration();
            });
    }
    void ResultsManager::sortScore(SortType type)
    {
        if (empty())
            return;
        sort(files.begin(), files.end(), [type](const Result& a, const Result& b) {
            if (a.getScore() == b.getScore()) {
                if (type == SortType::ASC)
                    return a.getDate() < b.getDate();
                return a.getDate() > b.getDate();
            }
            if (type == SortType::ASC)
                return a.getScore() < b.getScore();
            return a.getScore() > b.getScore();
            });
    }

    void ResultsManager::sortResults(SortField field, SortType type)
    {
        switch (field) {
            case SortField::DATE:
                sortDate(type);
                break;
            case SortField::MODEL:
                sortModel(type);
                break;
            case SortField::SCORE:
                sortScore(type);
                break;
            case SortField::DURATION:
                sortDuration(type);
                break;
        }
    }
    bool ResultsManager::empty() const
    {
        return files.empty();
    }
}