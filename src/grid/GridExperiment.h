#ifndef GRIDEXPERIMENT_H
#define GRIDEXPERIMENT_H
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
    class GridExperiment {
    public:
        explicit GridExperiment(struct ConfigGrid& config);
        void go(struct ConfigMPI& config_mpi);
        ~GridExperiment() = default;
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