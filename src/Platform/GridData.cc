#include "GridData.h"
#include <fstream>

namespace platform {
    GridData::GridData(const std::string& fileName)
    {
        json grid_file;
        std::ifstream resultData(fileName);
        if (resultData.is_open()) {
            grid_file = json::parse(resultData);
        } else {
            throw std::invalid_argument("Unable to open input file. [" + fileName + "]");
        }
        for (const auto& item : grid_file.items()) {
            auto key = item.key();
            auto value = item.value();
            grid[key] = value;
        }

    }
    int GridData::computeNumCombinations(const json& line)
    {
        int numCombinations = 1;
        for (const auto& item : line.items()) {
            numCombinations *= item.value().size();
        }
        return numCombinations;
    }
    int GridData::getNumCombinations(const std::string& dataset)
    {
        int numCombinations = 0;
        auto selected = decide_dataset(dataset);
        for (const auto& line : grid.at(selected)) {
            numCombinations += computeNumCombinations(line);
        }
        return numCombinations;
    }
    json GridData::generateCombinations(json::iterator index, const json::iterator last, std::vector<json>& output, json currentCombination)
    {
        if (index == last) {
            // If we reached the end of input, store the current combination
            output.push_back(currentCombination);
            return  currentCombination;
        }
        const auto& key = index.key();
        const auto& values = index.value();
        for (const auto& value : values) {
            auto combination = currentCombination;
            combination[key] = value;
            json::iterator nextIndex = index;
            generateCombinations(++nextIndex, last, output, combination);
        }
        return currentCombination;
    }
    std::vector<json> GridData::getGrid(const std::string& dataset)
    {
        auto selected = decide_dataset(dataset);
        auto result = std::vector<json>();
        for (json line : grid.at(selected)) {
            generateCombinations(line.begin(), line.end(), result, json({}));
        }
        return result;
    }
    json& GridData::getInputGrid(const std::string& dataset)
    {
        auto selected = decide_dataset(dataset);
        return grid.at(selected);
    }
    std::string GridData::decide_dataset(const std::string& dataset)
    {
        if (grid.find(dataset) != grid.end())
            return dataset;
        return ALL_DATASETS;
    }
} /* namespace platform */