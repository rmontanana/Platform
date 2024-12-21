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
    protected:
        virtual json build_tasks() = 0;
        virtual void save(json& results) = 0;
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