#ifndef GRIDBASE_H
#define GRIDBASE_H
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
    class GridBase {
    public:
        explicit GridBase(struct ConfigGrid& config)
        {
            this->config = config;
            if (config.smooth_strategy == "ORIGINAL")
                smooth_type = bayesnet::Smoothing_t::ORIGINAL;
            else if (config.smooth_strategy == "LAPLACE")
                smooth_type = bayesnet::Smoothing_t::LAPLACE;
            else if (config.smooth_strategy == "CESTNIK")
                smooth_type = bayesnet::Smoothing_t::CESTNIK;
            else {
                std::cerr << "GridBase: Unknown smoothing strategy: " << config.smooth_strategy << std::endl;
                exit(1);
            }
        };
        ~GridBase() = default;
        virtual void go(struct ConfigMPI& config_mpi) = 0;
    protected:
        virtual json build_tasks() = 0;
        struct ConfigGrid config;
        Timer timer; // used to measure the time of the whole process
        const std::string separator = "|";
        bayesnet::Smoothing_t smooth_type{ bayesnet::Smoothing_t::NONE };
    };
    class MPI_Base {
    public:
        static std::string get_color_rank(int rank)
        {
            auto colors = { Colors::WHITE(), Colors::RED(), Colors::GREEN(),  Colors::BLUE(), Colors::MAGENTA(), Colors::CYAN(), Colors::YELLOW(), Colors::BLACK() };
            std::string id = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
            auto idx = rank % id.size();
            return *(colors.begin() + rank % colors.size()) + id[idx];
        }
    };
} /* namespace platform */
#endif