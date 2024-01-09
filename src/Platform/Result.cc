#include "Result.h"
#include "BestScore.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include "Colors.h"
#include "DotEnv.h"
#include "CLocale.h"

namespace platform {
    Result::Result(const std::string& path, const std::string& filename)
        : path(path)
        , filename(filename)
    {
        auto data = load();
        date = data["date"];
        score = 0;
        for (const auto& result : data["results"]) {
            score += result["score"].get<double>();
        }
        scoreName = data["score_name"];
        auto best = BestScore::getScore(scoreName);
        if (best.first != "") {
            score /= best.second;
        }
        title = data["title"];
        duration = data["duration"];
        model = data["model"];
        complete = data["results"].size() > 1;
    }

    json Result::load() const
    {
        std::ifstream resultData(path + "/" + filename);
        if (resultData.is_open()) {
            json data = json::parse(resultData);
            return data;
        }
        throw std::invalid_argument("Unable to open result file. [" + path + "/" + filename + "]");
    }

    std::string Result::to_string(int maxModel) const
    {
        auto tmp = ConfigLocale();
        std::stringstream oss;
        double durationShow = duration > 3600 ? duration / 3600 : duration > 60 ? duration / 60 : duration;
        std::string durationUnit = duration > 3600 ? "h" : duration > 60 ? "m" : "s";
        oss << date << " ";
        oss << std::setw(maxModel) << std::left << model << " ";
        oss << std::setw(11) << std::left << scoreName << " ";
        oss << std::right << std::setw(11) << std::setprecision(7) << std::fixed << score << " ";
        auto completeString = isComplete() ? "C" : "P";
        oss << std::setw(1) << " " << completeString << "  ";
        oss << std::setw(7) << std::setprecision(2) << std::fixed << durationShow << " " << durationUnit << " ";
        oss << std::setw(50) << std::left << title << " ";
        return  oss.str();
    }
}