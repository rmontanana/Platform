#ifndef GRIDDATA_H
#define GRIDDATA_H
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

namespace platform {
    using json = nlohmann::json;
    const std::string ALL_DATASETS = "all";
    class GridData {
    public:
        explicit GridData(const std::string& fileName);
        ~GridData() = default;
        std::vector<json> getGrid(const std::string& dataset = ALL_DATASETS);
        int getNumCombinations(const std::string& dataset = ALL_DATASETS);
        json& getInputGrid(const std::string& dataset = ALL_DATASETS);
        std::map<std::string, json>& getGridFile() { return grid; }
    private:
        std::string decide_dataset(const std::string& dataset);
        json generateCombinations(json::iterator index, const json::iterator last, std::vector<json>& output, json currentCombination);
        int computeNumCombinations(const json& line);
        std::map<std::string, json> grid;
    };
} /* namespace platform */
#endif /* GRIDDATA_H */