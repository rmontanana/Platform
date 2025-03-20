#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H
#include <string>
#include <map>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <folding.hpp>
#include "common/Datasets.h"
#include "common/Timer.hpp"
#include "main/HyperParameters.h"
#include "GridData.h"
#include "GridBase.h"
#include "bayesnet/network/Network.h"


namespace platform {
    using json = nlohmann::ordered_json;
    class GridSearch : public GridBase {
    public:
        explicit GridSearch(struct ConfigGrid& config);
        ~GridSearch() = default;
        json loadResults();
        static inline std::string NO_CONTINUE() { return "NO_CONTINUE"; }
    private:
        void save(json& results);
        json initializeResults();
        std::vector<std::string> filterDatasets(Datasets& datasets) const;
        void compile_results(json& results, json& all_results, std::string& model);
        json store_result(std::vector<std::string>& names, Task_Result& result, json& results);
        void consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result);
    };
} /* namespace platform */
#endif