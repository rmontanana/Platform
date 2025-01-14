#ifndef GRIDEXPERIMENT_H
#define GRIDEXPERIMENT_H
#include <string>
#include <map>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include "common/Datasets.h"
#include "main/HyperParameters.h"
#include "GridData.h"
#include "GridBase.h"
#include "bayesnet/network/Network.h"


namespace platform {
    using json = nlohmann::ordered_json;
    class GridExperiment : public GridBase {
    public:
        explicit GridExperiment(struct ConfigGrid& config);
        ~GridExperiment() = default;
        json getResults();
    private:
        json computed_results;
        void save(json& results);
        json initializeResults();
        json build_tasks(Datasets& datasets);
        std::vector<std::string> filterDatasets(Datasets& datasets) const;
        void compile_results(json& results, json& all_results, std::string& model);
        json store_result(std::vector<std::string>& names, Task_Result& result, json& results);
        void consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result);
    };
} /* namespace platform */
#endif