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
        double score; // Experiment: Score test, no score train in this case
        double time; // Experiment: Time test
        double time_train;
        double nodes; // Experiment specific
        double leaves; // Experiment specific
        double depth; // Experiment specific
        int process;
        int task;
    } Task_Result;
    const int TAG_QUERY = 1;
    const int TAG_RESULT = 2;
    const int TAG_TASK = 3;
    const int TAG_END = 4;
} /* namespace platform */
#endif