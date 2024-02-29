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
    public:
        DotEnv()
        {
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
                    env[key] = value;
                }
            }
        }
        std::string get(const std::string& key)
        {
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