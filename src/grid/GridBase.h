#ifndef GRIDBASE_H
#define GRIDBASE_H
#include <string>
#include <map>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include "common/Datasets.h"
#include "common/Timer.h"
#include "common/Colors.h"
#include "main/HyperParameters.h"
#include "GridData.h"
#include "GridConfig.h"
#include "bayesnet/network/Network.h"


namespace platform {
    using json = nlohmann::ordered_json;
    class GridBase {
    public:
        explicit GridBase(struct ConfigGrid& config);
        ~GridBase() = default;
        void go(struct ConfigMPI& config_mpi);
        void validate_config();
    protected:
        virtual json build_tasks(Datasets& datasets) = 0;
        virtual void save(json& results) = 0;
        virtual std::vector<std::string> filterDatasets(Datasets& datasets) const = 0;
        virtual json initializeResults() = 0;
        virtual void compile_results(json& results, json& all_results, std::string& model) = 0;
        virtual json store_result(std::vector<std::string>& names, Task_Result& result, json& results) = 0;
        virtual void consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result) = 0;
        void shuffle_and_progress_bar(json& tasks);
        json producer(std::vector<std::string>& names, json& tasks, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result);
        void consumer(Datasets& datasets, json& tasks, struct ConfigGrid& config, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result);
        std::string get_color_rank(int rank);
        void summary(json& all_results, json& tasks, struct ConfigMPI& config_mpi);
        struct ConfigGrid config;
        Timer timer; // used to measure the time of the whole process
        const std::string separator = "|";
        bayesnet::Smoothing_t smooth_type{ bayesnet::Smoothing_t::NONE };
    };
} /* namespace platform */
#endif