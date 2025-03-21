#include <iostream>
#include <torch/torch.h>
#include <folding.hpp>
#include "main/Models.h"
#include "common/Paths.h"
#include "common/Utils.h"
#include "common/Colors.h"
#include "GridSearch.h"

namespace platform {
    GridSearch::GridSearch(struct ConfigGrid& config) : GridBase(config)
    {
    }
    json GridSearch::loadResults()
    {
        std::ifstream file(Paths::grid_output(config.model));
        if (file.is_open()) {
            return json::parse(file);
        }
        return json();
    }
    std::vector<std::string> GridSearch::filterDatasets(Datasets& datasets) const
    {
        // Load datasets
        auto datasets_names = datasets.getNames();
        if (config.continue_from != NO_CONTINUE()) {
            // Continue previous execution:
            if (std::find(datasets_names.begin(), datasets_names.end(), config.continue_from) == datasets_names.end()) {
                throw std::invalid_argument("Dataset " + config.continue_from + " not found");
            }
            // Remove datasets already processed
            std::vector<string>::iterator it = datasets_names.begin();
            while (it != datasets_names.end()) {
                if (*it != config.continue_from) {
                    it = datasets_names.erase(it);
                } else {
                    if (config.only)
                        ++it;
                    else
                        break;
                }
            }
        }
        // Exclude datasets
        for (const auto& name : config.excluded) {
            auto dataset = name.get<std::string>();
            auto it = std::find(datasets_names.begin(), datasets_names.end(), dataset);
            if (it == datasets_names.end()) {
                throw std::invalid_argument("Dataset " + dataset + " already excluded or doesn't exist!");
            }
            datasets_names.erase(it);
        }
        return datasets_names;
    }
    json GridSearch::initializeResults()
    {
        // Load previous results if continue is set
        json results;
        if (config.continue_from != NO_CONTINUE()) {
            if (!config.quiet)
                std::cout << Colors::RESET() << "* Loading previous results" << std::endl;
            try {
                std::ifstream file(Paths::grid_output(config.model));
                if (file.is_open()) {
                    results = json::parse(file);
                    results = results["results"];
                }
            }
            catch (const std::exception& e) {
                std::cerr << "* There were no previous results" << std::endl;
                std::cerr << "* Initizalizing new results" << std::endl;
                results = json();
            }
        }
        return results;
    }
    void GridSearch::save(json& results)
    {
        std::ofstream file(Paths::grid_output(config.model));
        json output = {
            { "model", config.model },
            { "score", config.score },
            { "discretize", config.discretize },
            { "stratified", config.stratified },
            { "n_folds", config.n_folds },
            { "seeds", config.seeds },
            { "date", get_date() + " " + get_time()},
            { "nested", config.nested},
            { "platform", config.platform },
            { "duration", timer.getDurationString(true)},
            { "results", results }

        };
        file << output.dump(4);
    }
    void GridSearch::compile_results(json& results, json& all_results, std::string& model)
    {
        Timer timer;
        auto grid = GridData(Paths::grid_input(model));
        //
        // Select the best result of the computed outer folds
        //
        for (const auto& result : all_results.items()) {
            // each result has the results of all the outer folds as each one were a different task
            double best_score = 0.0;
            json best;
            for (const auto& result_fold : result.value()) {
                double score = result_fold["score"].get<double>();
                if (score > best_score) {
                    best_score = score;
                    best = result_fold;
                }
            }
            auto dataset = result.key();
            auto combinations = grid.getGrid(dataset);
            json json_best = {
                    { "score", best_score },
                    { "hyperparameters", combinations[best["combination"].get<int>()] },
                    { "date", get_date() + " " + get_time() },
                    { "grid", grid.getInputGrid(dataset) },
                    { "duration", timer.translate2String(best["time"].get<double>()) }
            };
            results[dataset] = json_best;
        }
    }
    json GridSearch::store_result(std::vector<std::string>& names, Task_Result& result, json& results)
    {
        json json_result = {
            { "score", result.score },
            { "combination", result.idx_combination },
            { "fold", result.n_fold },
            { "time", result.time },
            { "dataset", result.idx_dataset },
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
    void GridSearch::consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result)
    {
        //
        // initialize
        //
        Timer timer;
        timer.start();
        json task = tasks[n_task];
        auto model = config.model;
        auto grid = GridData(Paths::grid_input(model));
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
        auto combinations = grid.getGrid(dataset_name);
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
        float best_fold_score = 0.0;
        int best_idx_combination = -1;
        json best_fold_hyper;
        for (int idx_combination = 0; idx_combination < combinations.size(); ++idx_combination) {
            auto hyperparam_line = combinations[idx_combination];
            auto hyperparameters = platform::HyperParameters(datasets.getNames(), hyperparam_line);
            folding::Fold* nested_fold;
            if (config.stratified)
                nested_fold = new folding::StratifiedKFold(config.nested, y_train, seed);
            else
                nested_fold = new folding::KFold(config.nested, y_train.size(0), seed);
            double score = 0.0;
            for (int n_nested_fold = 0; n_nested_fold < config.nested; n_nested_fold++) {
                //
                // Nested level fold
                //
                auto [train_nested, test_nested] = nested_fold->getFold(n_nested_fold);
                auto train_nested_t = torch::tensor(train_nested);
                auto test_nested_t = torch::tensor(test_nested);
                auto X_nested_train = X_train.index({ "...", train_nested_t });
                auto y_nested_train = y_train.index({ train_nested_t });
                auto X_nested_test = X_train.index({ "...", test_nested_t });
                auto y_nested_test = y_train.index({ test_nested_t });
                //
                // Build Classifier with selected hyperparameters
                //
                auto clf = Models::instance()->create(config.model);
                auto valid = clf->getValidHyperparameters();
                hyperparameters.check(valid, dataset_name);
                clf->setHyperparameters(hyperparameters.get(dataset_name));
                //
                // Train model
                //
                clf->fit(X_nested_train, y_nested_train, features, className, states, smooth);
                //
                // Test model
                //
                score += clf->score(X_nested_test, y_nested_test);
            }
            delete nested_fold;
            score /= config.nested;
            if (score > best_fold_score) {
                best_fold_score = score;
                best_idx_combination = idx_combination;
                best_fold_hyper = hyperparam_line;
            }
        }
        delete fold;
        //
        // Build Classifier with the best hyperparameters to obtain the best score
        //
        auto hyperparameters = platform::HyperParameters(datasets.getNames(), best_fold_hyper);
        auto clf = Models::instance()->create(config.model);
        auto valid = clf->getValidHyperparameters();
        hyperparameters.check(valid, dataset_name);
        clf->setHyperparameters(best_fold_hyper);
        clf->fit(X_train, y_train, features, className, states, smooth);
        best_fold_score = clf->score(X_test, y_test);
        //
        // Return the result
        //
        result->idx_dataset = task["idx_dataset"].get<int>();
        result->idx_combination = best_idx_combination;
        result->score = best_fold_score;
        result->n_fold = n_fold;
        result->time = timer.getDuration();
        result->process = config_mpi.rank;
        result->task = n_task;
        //
        // Update progress bar
        //
        std::cout << get_color_rank(config_mpi.rank) << std::flush;
    }
} /* namespace platform */
