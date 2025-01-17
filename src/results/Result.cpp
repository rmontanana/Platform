#include <filesystem>
#include <fstream>
#include <sstream>
#include "best/BestScore.h"
#include "common/Colors.h"
#include "common/DotEnv.h"
#include "common/CLocale.h"
#include "common/Paths.h"
#include "common/Symbols.h"
#include "Result.h"
#include "JsonValidator.h"
#include "SchemaV1_0.h"

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
    std::vector<std::string> Result::check()
    {
        platform::JsonValidator validator(platform::SchemaV1_0::schema);
        return validator.validate(data);
    }
    void Result::save()
    {
        std::ofstream file(Paths::results() + getFilename());
        file << data;
        file.close();
    }
    std::string Result::getFilename() const
    {
        std::ostringstream oss;
        std::string stratified;
        try {
            stratified = data["stratified"].get<bool>() ? "1" : "0";
        }
        catch (nlohmann::json_abi_v3_11_3::detail::type_error) {
            stratified = data["stratified"].get<int>() == 1 ? "1" : "0";
        }
        oss << "results_"
            << data.at("score_name").get<std::string>() << "_"
            << data.at("model").get<std::string>() << "_"
            << data.at("platform").get<std::string>() << "_"
            << data["date"].get<std::string>() << "_"
            << data["time"].get<std::string>() << "_"
            << stratified << ".json";
        return oss.str();
    }
    std::string Result::to_string(int maxModel, int maxTitle) const
    {
        auto tmp = ConfigLocale();
        std::stringstream oss;
        std::string s, d;
        try {
            s = data["stratified"].get<bool>() ? "S" : " ";
        }
        catch (nlohmann::json_abi_v3_11_3::detail::type_error) {
            s = data["stratified"].get<int>() == 1 ? "S" : " ";
        }
        try {
            d = data["discretized"].get<bool>() ? "D" : " ";
        }
        catch (nlohmann::json_abi_v3_11_3::detail::type_error) {
            d = data["discretized"].get<int>() == 1 ? "D" : " ";
        }
        auto duration = data["duration"].get<double>();
        double durationShow = duration > 3600 ? duration / 3600 : duration > 60 ? duration / 60 : duration;
        std::string durationUnit = duration > 3600 ? "h" : duration > 60 ? "m" : "s";
        oss << data["date"].get<std::string>() << " ";
        oss << std::setw(maxModel) << std::left << data["model"].get<std::string>() << " ";
        oss << std::setw(11) << std::left << data["score_name"].get<std::string>() << " ";
        oss << std::right << std::setw(10) << std::setprecision(7) << std::fixed << score << " ";
        oss << std::left << std::setw(12) << data["platform"].get<std::string>() << " ";
        oss << s << d << " ";
        auto completeString = isComplete() ? "C" : "P";
        oss << std::setw(1) << " " << completeString << "  ";
        oss << std::setw(5) << std::right << std::setprecision(2) << std::fixed << durationShow << " " << durationUnit << " ";
        auto title = data["title"].get<std::string>();
        if (title.size() > maxTitle) {
            title = title.substr(0, maxTitle - 1) + Symbols::ellipsis;
        }
        oss << std::setw(maxTitle) << std::left << title;
        return  oss.str();
    }
}