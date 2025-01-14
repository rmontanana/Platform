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
        json loadResults();
        static inline std::string NO_CONTINUE() { return "NO_CONTINUE"; }
    private:
        void save(json& results);
        json initializeResults();
        std::vector<std::string> filterDatasets(Datasets& datasets) const;
        json producer(std::vector<std::string>& names, json& tasks, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result);
        void consumer(Datasets& datasets, json& tasks, struct ConfigGrid& config, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result);
        void select_best_results_folds(json& results, json& all_results, std::string& model);
        json store_result(std::vector<std::string>& names, Task_Result& result, json& results);
        void consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result);
    };
} /* namespace platform */
#endif