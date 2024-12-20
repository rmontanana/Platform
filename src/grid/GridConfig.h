#ifndef GRIDCONFIG_H
#define GRIDCONFIG_H
#include <string>
#include <map>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include "common/Datasets.h"
#include "common/Timer.h"
#include "main/HyperParameters.h"
#include "GridData.h"
#include "GridConfig.h"
#include "bayesnet/network/Network.h"


namespace platform {
    using json = nlohmann::ordered_json;
    struct ConfigGrid {
        std::string model;
        std::string score;
        std::string continue_from;
        std::string platform;
        std::string smooth_strategy;
        bool quiet;
        bool only; // used with continue_from to only compute that dataset
        bool discretize;
        bool stratified;
        int nested;
        int n_folds;
        json excluded;
        std::vector<int> seeds;
    };
    struct ConfigMPI {
        int rank;
        int n_procs;
        int manager;
    };
    typedef struct {
        uint idx_dataset;
        uint idx_combination;
        int n_fold;
        double score;
        double time;
    } Task_Result;
    const int TAG_QUERY = 1;
    const int TAG_RESULT = 2;
    const int TAG_TASK = 3;
    const int TAG_END = 4;
    /* *************************************************************************************************************
    //
    // MPI Common Functions
    //
    ************************************************************************************************************* */
    std::string get_color_rank(int rank);
    /* *************************************************************************************************************
    //
    // MPI Experiment Functions
    //
    ************************************************************************************************************* */
    json mpi_experiment_producer(std::vector<std::string>& names, json& tasks, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result);
    void mpi_experiment_consumer(Datasets& datasets, json& tasks, struct ConfigGrid& config, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result);
    void join_results_folds(json& results, json& all_results, std::string& model);
    json store_experiment_result(std::vector<std::string>& names, Task_Result& result, json& results);
    void mpi_experiment_consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tass, int n_task, Datasets& datasets, Task_Result* result);
    /* *************************************************************************************************************
    //
    // MPI Search Functions
    //
    ************************************************************************************************************* */
    json mpi_search_producer(std::vector<std::string>& names, json& tasks, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result);
    void mpi_search_consumer(Datasets& datasets, json& tasks, struct ConfigGrid& config, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result);
    void select_best_results_folds(json& results, json& all_results, std::string& model);
    json store_search_result(std::vector<std::string>& names, Task_Result& result, json& results);
    void mpi_search_consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result);
} /* namespace platform */
#endif