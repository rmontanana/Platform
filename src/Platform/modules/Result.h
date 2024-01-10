#ifndef RESULT_H
#define RESULT_H
#include <map>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
namespace platform {
    using json = nlohmann::json;

    class Result {
    public:
        Result(const std::string& path, const std::string& filename);
        json load() const;
        std::string to_string(int maxModel) const;
        std::string getFilename() const { return filename; };
        std::string getDate() const { return date; };
        double getScore() const { return score; };
        std::string getTitle() const { return title; };
        double getDuration() const { return duration; };
        std::string getModel() const { return model; };
        std::string getScoreName() const { return scoreName; };
        bool isComplete() const { return complete; };
    private:
        std::string path;
        std::string filename;
        std::string date;
        double score;
        std::string title;
        double duration;
        std::string model;
        std::string scoreName;
        bool complete;
    };
};
#endif