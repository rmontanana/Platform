#ifndef GRIDEXPERIMENT_H
#define GRIDEXPERIMENT_H
#include <string>
#include <map>
#include <mpi.h>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "common/Datasets.h"
#include "common/DotEnv.h"
#include "main/Experiment.h"
#include "main/HyperParameters.h"
#include "main/ArgumentsExperiment.h"
#include "GridData.h"
#include "GridBase.h"
#include "bayesnet/network/Network.h"


namespace platform {
    using json = nlohmann::ordered_json;
    class GridExperiment : public GridBase {
    public:
        explicit GridExperiment(ArgumentsExperiment& program, struct ConfigGrid& config);
        ~GridExperiment() = default;
        json getResults();
        Experiment& getExperiment() { return experiment; }
        size_t numFiles() const { return filesToTest.size(); }
        bool haveToSaveResults() const { return saveResults; }
    private:
        ArgumentsExperiment& arguments;
        Experiment experiment;
        json computed_results;
        bool saveResults = false;
        std::vector<std::string> filesToTest;
        void save(json& results);
        json initializeResults();
        std::vector<std::string> filterDatasets(Datasets& datasets) const;
        void compile_results(json& results, json& all_results, std::string& model);
        json store_result(std::vector<std::string>& names, Task_Result& result, json& results);
        void consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result);
    };
} /* namespace platform */
#endif