#include <argparse/argparse.hpp>
#include "main/Experiment.h"
#include "main/ArgumentsExperiment.h"
#include "config_platform.h"


using json = nlohmann::ordered_json;


int main(int argc, char** argv)
{
    argparse::ArgumentParser program("b_main", { platform_project_version.begin(), platform_project_version.end() });
    auto arguments = platform::ArgumentsExperiment(program, platform::experiment_t::NORMAL);
    arguments.add_arguments();
    arguments.parse_args(argc, argv);
    /*
     * Begin Processing
     */
     // Initialize the experiment class with the command line arguments
    auto experiment = arguments.initializedExperiment();
    auto path_results = arguments.getPathResults();
    platform::Timer timer;
    timer.start();
    experiment.go();
    experiment.setDuration(timer.getDuration());
    if (!arguments.isQuiet()) {
        // Classification report if only one dataset is tested
        experiment.report();
    }
    if (arguments.haveToSaveResults()) {
        experiment.saveResult(path_results);
    }
    if (arguments.doGraph()) {
        experiment.saveGraph();
    }
    return 0;
}
