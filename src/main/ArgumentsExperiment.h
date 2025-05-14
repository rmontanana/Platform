#ifndef ARGUMENTSEXPERIMENT_H
#define ARGUMENTSEXPERIMENT_H
#include <string>
#include <iostream>
#include <vector>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "Experiment.h"

namespace platform {
    using json = nlohmann::ordered_json;
    enum class experiment_t { NORMAL, GRID };
    class ArgumentsExperiment {
    public:
        ArgumentsExperiment(argparse::ArgumentParser& program, experiment_t type);
        ~ArgumentsExperiment() = default;
        std::vector<std::string> getFilesToTest() const { return filesToTest; }
        void add_arguments();
        void parse_args(int argc, char** argv);
        void parse();
        Experiment& initializedExperiment();
        bool isQuiet() const { return quiet; }
        bool haveToSaveResults() const { return saveResults; }
        bool doGraph() const { return graph; }
        std::string getPathResults() const { return path_results; }
    private:
        Experiment experiment;
        experiment_t type;
        argparse::ArgumentParser& arguments;
        std::string file_name, model_name, title, hyperparameters_file, datasets_file, discretize_algo, smooth_strat;
        std::string score, path_results;
        json hyperparameters_json;
        bool discretize_dataset, stratified, saveResults, quiet, no_train_score, generate_fold_files, graph, hyper_best;
        std::vector<int> seeds;
        std::vector<std::string> file_names;
        std::vector<std::string> filesToTest;
        platform::HyperParameters test_hyperparams;
        int n_folds;
    };
}
#endif