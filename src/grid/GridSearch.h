#ifndef GRIDSEARCH_H
#define GRIDSEARCH_H
#include <string>
#include <map>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include "common/Datasets.h"
#include "common/Timer.h"
#include "main/HyperParameters.h"
#include "GridData.h"
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
    class GridSearch {
    public:
        explicit GridSearch(struct ConfigGrid& config);
        void go(struct ConfigMPI& config_mpi);
        ~GridSearch() = default;
        json loadResults();
        static inline std::string NO_CONTINUE() { return "NO_CONTINUE"; }
    private:
        void save(json& results);
        json initializeResults();
        std::vector<std::string> filterDatasets(Datasets& datasets) const;
        struct ConfigGrid config;
        json build_tasks_mpi(int rank);
        Timer timer; // used to measure the time of the whole process
        const std::string separator = "|";
        bayesnet::Smoothing_t smooth_type{ bayesnet::Smoothing_t::NONE };
    };
} /* namespace platform */
#endif