#ifndef DOTENV_H
#define DOTENV_H
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include "Utils.h"

//#include "Dataset.h"
namespace platform {
    class DotEnv {
    private:
        std::map<std::string, std::string> env;
        std::map<std::string, std::vector<std::string>> valid;
    public:
        DotEnv(bool create = false)
        {
            valid =
            {
                {"source_data", {"Arff", "Tanveer", "Surcov"}},
                {"experiment", {"discretiz", "odte", "covid"}},
                {"fit_features", {"0", "1"}},
                {"discretize", {"0", "1"}},
                {"ignore_nan", {"0", "1"}},
                {"stratified", {"0", "1"}},
                {"score", {"accuracy"}},
                {"framework", {"bulma", "bootstrap"}},
                {"margin", {"0.1", "0.2", "0.3"}},
                {"n_folds", {"5", "10"}},
                {"discretize_algo", {"mdlp", "bin3u", "bin3q", "bin4u", "bin4q"}},
                {"platform", {"any"}},
                {"model", {"any"}},
                {"seeds", {"any"}},
                {"nodes", {"any"}},
                {"leaves", {"any"}},
                {"depth", {"any"}},
            };
            if (create) {
                // For testing purposes
                std::ofstream file(".env");
                file << "source_data = Test" << std::endl;
                file << "margin = 0.1" << std::endl;
                file.close();
            }
            std::ifstream file(".env");
            if (!file.is_open()) {
                std::cerr << "File .env not found" << std::endl;
                exit(1);
            }
            std::string line;
            while (std::getline(file, line)) {
                line = trim(line);
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                std::istringstream iss(line);
                std::string key, value;
                if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                    key = trim(key);
                    value = trim(value);
                    parse(key, value);
                    env[key] = value;
                }
            }
            parseEnv();
        }
        void parse(const std::string& key, const std::string& value)
        {
            if (valid.find(key) == valid.end()) {
                std::cerr << "Invalid key in .env: " << key << std::endl;
                exit(1);
            }
            if (valid[key].front() == "any") {
                return;
            }
            if (std::find(valid[key].begin(), valid[key].end(), value) == valid[key].end()) {
                std::cerr << "Invalid value in .env: " << key << " = " << value << std::endl;
                exit(1);
            }
        }
        void parseEnv()
        {
            for (auto& [key, values] : valid) {
                if (env.find(key) == env.end()) {
                    std::string valid_values = "", sep = "";
                    for (const auto& value : values) {
                        valid_values += sep + value;
                        sep = ", ";
                    }
                    std::cerr << "Key not found in .env: " << key << ", valid values: " << valid_values << std::endl;
                    exit(1);
                }
            }
        }
        std::string get(const std::string& key)
        {
            if (env.find(key) == env.end()) {
                std::cerr << "Key not found in .env: " << key << std::endl;
                exit(1);
            }
            return env.at(key);
        }
        std::vector<int> getSeeds()
        {
            auto seeds = std::vector<int>();
            auto seeds_str = env["seeds"];
            seeds_str = trim(seeds_str);
            seeds_str = seeds_str.substr(1, seeds_str.size() - 2);
            auto seeds_str_split = split(seeds_str, ',');
            transform(seeds_str_split.begin(), seeds_str_split.end(), back_inserter(seeds), [](const std::string& str) {
                return stoi(str);
                });
            return seeds;
        }
    };
}
#endif