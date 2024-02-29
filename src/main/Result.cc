#include "Result.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include "BestScore.h"
#include "Colors.h"
#include "DotEnv.h"
#include "CLocale.h"
#include "Paths.h"

namespace platform {
    std::string get_actual_date()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%Y-%m-%d");
        return oss.str();
    }
    std::string get_actual_time()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%H:%M:%S");
        return oss.str();
    }
    Result::Result()
    {
        data["date"] = get_actual_date();
        data["time"] = get_actual_time();
        data["results"] = json::array();
        data["seeds"] = json::array();
    }

    Result& Result::load(const std::string& path, const std::string& fileName)
    {
        std::ifstream resultData(path + "/" + fileName);
        if (resultData.is_open()) {
            data = json::parse(resultData);
        } else {
            throw std::invalid_argument("Unable to open result file. [" + path + "/" + fileName + "]");
        }
        score = 0;
        for (const auto& result : data["results"]) {
            score += result["score"].get<double>();
        }
        auto scoreName = data["score_name"];
        auto best = BestScore::getScore(scoreName);
        if (best.first != "") {
            score /= best.second;
        }
        complete = data["results"].size() > 1;
        return *this;
    }
    json Result::getJson()
    {
        return data;
    }

    void Result::save()
    {
        std::ofstream file(Paths::results() + "/" + getFilename());
        file << data;
        file.close();
    }
    std::string Result::getFilename() const
    {
        std::ostringstream oss;
        oss << "results_" << data.at("score_name").get<std::string>() << "_" << data.at("model").get<std::string>() << "_"
            << data.at("platform").get<std::string>() << "_" << data["date"].get<std::string>() << "_"
            << data["time"].get<std::string>() << "_" << (data["stratified"] ? "1" : "0") << ".json";
        return oss.str();
    }


    std::string Result::to_string(int maxModel) const
    {
        auto tmp = ConfigLocale();
        std::stringstream oss;
        auto duration = data["duration"].get<double>();
        double durationShow = duration > 3600 ? duration / 3600 : duration > 60 ? duration / 60 : duration;
        std::string durationUnit = duration > 3600 ? "h" : duration > 60 ? "m" : "s";
        oss << data["date"].get<std::string>() << " ";
        oss << std::setw(maxModel) << std::left << data["model"].get<std::string>() << " ";
        oss << std::setw(11) << std::left << data["score_name"].get<std::string>() << " ";
        oss << std::right << std::setw(11) << std::setprecision(7) << std::fixed << score << " ";
        auto completeString = isComplete() ? "C" : "P";
        oss << std::setw(1) << " " << completeString << "  ";
        oss << std::setw(7) << std::setprecision(2) << std::fixed << durationShow << " " << durationUnit << " ";
        oss << std::setw(50) << std::left << data["title"].get<std::string>() << " ";
        return  oss.str();
    }
}