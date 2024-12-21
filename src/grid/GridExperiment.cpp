#include <iostream>
#include <cstddef>
#include <torch/torch.h>
#include <folding.hpp>
#include "main/Models.h"
#include "common/Paths.h"
#include "common/Colors.h"
#include "common/Utils.h"
#include "GridExperiment.h"

namespace platform {

    GridExperiment::GridExperiment(struct ConfigGrid& config) : GridBase(config)
    {
    }
    json GridExperiment::loadResults()
    {
        std::ifstream file(Paths::grid_output(config.model));
        if (file.is_open()) {
            return json::parse(file);
        }
        return json();
    }
    json GridExperiment::build_tasks()
    {
        /*
        * Each task is a json object with the following structure:
        * {
        *   "dataset": "dataset_name",
        *   "idx_dataset": idx_dataset, // used to identify the dataset in the results
        *    // this index is relative to the list of used datasets in the actual run not to the whole datasets list
        *   "seed": # of seed to use,
        *   "fold": # of fold to process
        * }
        */
        auto tasks = json::array();
        auto grid = GridData(Paths::grid_input(config.model));
        auto datasets = Datasets(false, Paths::datasets());
        auto all_datasets = datasets.getNames();
        auto datasets_names = all_datasets;
        for (int idx_dataset = 0; idx_dataset < datasets_names.size(); ++idx_dataset) {
            auto dataset = datasets_names[idx_dataset];
            for (const auto& seed : config.seeds) {
                auto combinations = grid.getGrid(dataset);
                for (int n_fold = 0; n_fold < config.n_folds; n_fold++) {
                    json task = {
                        { "dataset", dataset },
                        { "idx_dataset", idx_dataset},
                        { "seed", seed },
                        { "fold", n_fold},
                    };
                    tasks.push_back(task);
                }
            }
        }
        // Shuffle the array so heavy datasets are eas  ier spread across the workers
        std::mt19937 g{ 271 }; // Use fixed seed to obtain the same shuffle
        std::shuffle(tasks.begin(), tasks.end(), g);
        std::cout << "* Number of tasks: " << tasks.size() << std::endl;
        std::cout << separator << std::flush;
        for (int i = 0; i < tasks.size(); ++i) {
            if ((i + 1) % 10 == 0)
                std::cout << separator;
            else
                std::cout << (i + 1) % 10;
        }
        std::cout << separator << std::endl << separator << std::flush;
        return tasks;
    }
    json GridExperiment::initializeResults()
    {
        // Load previous results if continue is set
        json results;
        return results;
    }
    void GridExperiment::save(json& results)
    {
        std::ofstream file(Paths::grid_output(config.model));
        json output = {
            { "model", config.model },
            { "score", config.score },
            { "discretize", config.discretize },
            { "stratified", config.stratified },
            { "n_folds", config.n_folds },
            { "seeds", config.seeds },
            { "date", get_date() + " " + get_time()},
            { "nested", config.nested},
            { "platform", config.platform },
            { "duration", timer.getDurationString(true)},
            { "results", results }

        };
        file << output.dump(4);
    }

} /* namespace platform */