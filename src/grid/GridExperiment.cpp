#include <iostream>
#include <cstddef>
#include <torch/torch.h>
#include <folding.hpp>
#include "main/Models.h"
#include "common/Paths.h"
#include "common/Utils.h"
#include "GridExperiment.h"

namespace platform {
    GridExperiment::GridExperiment(struct ConfigGrid& config) : GridBase(config)
    {
    }
    json GridExperiment::getResults()
    {
        return computed_results;
    }
    json GridExperiment::build_tasks(Datasets& datasets)
    {
        /*
        * Each task is a json object with the following structure:
        * {
        *   "dataset": "dataset_name",
        *   "idx_dataset": idx_dataset, // used to identify the dataset in the results
        *    // this index is relative to the list of used datasets in the actual run not to the whole datasets list
        *   "seed": # of seed to use,
        *   "fold": # of fold to process
        *   "hyperpameters": json object with the hyperparameters to use
        * }
        * This way a task consists in process all combinations of hyperparameters for a dataset, seed and fold
        */
        auto tasks = json::array();
        auto all_datasets = datasets.getNames();
        auto datasets_names = filterDatasets(datasets);
        for (int idx_dataset = 0; idx_dataset < datasets_names.size(); ++idx_dataset) {
            auto dataset = datasets_names[idx_dataset];
            for (const auto& seed : config.seeds) {
                for (int n_fold = 0; n_fold < config.n_folds; n_fold++) {
                    json task = {
                        { "dataset", dataset },
                        { "idx_dataset", idx_dataset},
                        { "seed", seed },
                        { "fold", n_fold},
                        { "hyperparameters", json::object() }
                    };
                    tasks.push_back(task);
                }
            }
        }
        shuffle_and_progress_bar(tasks);
        return tasks;
    }
    std::vector<std::string> GridExperiment::filterDatasets(Datasets& datasets) const
    {
        // Load datasets
        auto datasets_names = datasets.getNames();
        datasets_names.clear();
        datasets_names.push_back("iris");
        return datasets_names;
    }
    json GridExperiment::initializeResults()
    {
        json results;
        return results;
    }
    void GridExperiment::save(json& results)
    {
        // std::ofstream file(Paths::grid_output(config.model));
        // json output = {
        //     { "model", config.model },
        //     { "score", config.score },
        //     { "discretize", config.discretize },
        //     { "stratified", config.stratified },
        //     { "n_folds", config.n_folds },
        //     { "seeds", config.seeds },
        //     { "date", get_date() + " " + get_time()},
        //     { "nested", config.nested},
        //     { "platform", config.platform },
        //     { "duration", timer.getDurationString(true)},
        //     { "results", results }
        // };
        // file << output.dump(4);
    }
    void GridExperiment::compile_results(json& results, json& all_results, std::string& model)
    {
        results = json::object();
        for (const auto& result : all_results.items()) {
            // each result has the results of all the outer folds as each one were a different task
            auto dataset = result.key();
            results[dataset] = json::array();
            for (int fold = 0; fold < result.value().size(); ++fold) {
                results[dataset].push_back(json::object());
            }
            for (const auto& result_fold : result.value()) {
                results[dataset][result_fold["fold"].get<int>()] = result_fold;
            }
        }
        computed_results = results;
    }
    json GridExperiment::store_result(std::vector<std::string>& names, Task_Result& result, json& results)
    {
        json json_result = {
            { "score", result.score },
            { "combination", result.idx_combination },
            { "fold", result.n_fold },
            { "time", result.time },
            { "dataset", result.idx_dataset },
            { "nodes", result.nodes },
            { "leaves", result.leaves },
            { "depth", result.depth },
            { "process", result.process },
            { "task", result.task }
        };
        auto name = names[result.idx_dataset];
        if (!results.contains(name)) {
            results[name] = json::array();
        }
        results[name].push_back(json_result);
        return results;
    }
    void GridExperiment::consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result)
    {
        //
        // initialize
        //
        Timer timer;
        timer.start();
        json task = tasks[n_task];
        auto model = config.model;
        auto dataset_name = task["dataset"].get<std::string>();
        auto idx_dataset = task["idx_dataset"].get<int>();
        auto seed = task["seed"].get<int>();
        auto n_fold = task["fold"].get<int>();
        bool stratified = config.stratified;
        bayesnet::Smoothing_t smooth;
        if (config.smooth_strategy == "ORIGINAL")
            smooth = bayesnet::Smoothing_t::ORIGINAL;
        else if (config.smooth_strategy == "LAPLACE")
            smooth = bayesnet::Smoothing_t::LAPLACE;
        else if (config.smooth_strategy == "CESTNIK")
            smooth = bayesnet::Smoothing_t::CESTNIK;
        //
        // Generate the hyperparameters combinations
        //
        auto& dataset = datasets.getDataset(dataset_name);
        dataset.load();
        auto [X, y] = dataset.getTensors();
        auto features = dataset.getFeatures();
        auto className = dataset.getClassName();
        //
        // Start working on task
        //
        folding::Fold* fold;
        if (stratified)
            fold = new folding::StratifiedKFold(config.n_folds, y, seed);
        else
            fold = new folding::KFold(config.n_folds, y.size(0), seed);
        auto [train, test] = fold->getFold(n_fold);
        auto [X_train, X_test, y_train, y_test] = dataset.getTrainTestTensors(train, test);
        auto states = dataset.getStates(); // Get the states of the features Once they are discretized

        //
        // Build Classifier with selected hyperparameters
        //
        auto clf = Models::instance()->create(config.model);
        auto valid = clf->getValidHyperparameters();
        auto hyperparameters = platform::HyperParameters(datasets.getNames(), task["hyperparameters"]);
        hyperparameters.check(valid, dataset_name);
        clf->setHyperparameters(hyperparameters.get(dataset_name));
        //
        // Train model
        //
        clf->fit(X_train, y_train, features, className, states, smooth);
        //
        // Test model
        //
        double score = clf->score(X_test, y_test);
        delete fold;
        //
        // Return the result
        //
        result->idx_dataset = task["idx_dataset"].get<int>();
        result->idx_combination = 0;
        result->score = score;
        result->n_fold = n_fold;
        result->time = timer.getDuration();
        result->nodes = clf->getNumberOfNodes();
        result->leaves = clf->getNumberOfEdges();
        result->depth = clf->getNumberOfStates();
        result->process = config_mpi.rank;
        result->task = n_task;
        //
        // Update progress bar
        //
        std::cout << get_color_rank(config_mpi.rank) << std::flush;
    }
} /* namespace platform */