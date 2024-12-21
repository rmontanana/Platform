#include "common/DotEnv.h"
#include "common/Paths.h"
#include "GridBase.h"

namespace platform {

    GridBase::GridBase(struct ConfigGrid& config)
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
    }

}