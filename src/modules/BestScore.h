#ifndef BESTSCORE_H
#define BESTSCORE_H
#include <string>
#include <map>
#include <utility>
#include "DotEnv.h"
namespace platform {
    class BestScore {
    public:
        static std::pair<std::string, double> getScore(const std::string& metric)
        {
            static std::map<std::pair<std::string, std::string>, std::pair<std::string, double>> data = {
               {{"discretiz", "accuracy"}, {"STree_default (linear-ovo)",  22.109799}},
               {{"odte", "accuracy"}, {"STree_default (linear-ovo)",  22.109799}},
            };
            auto env = platform::DotEnv();
            std::string experiment = env.get("experiment");
            try {
                return data[{experiment, metric}];
            }
            catch (...) {
                return { "", 0.0 };
            }
        }
    };
}

#endif