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
#include "GridBase.h"
#include "bayesnet/network/Network.h"


namespace platform {
    using json = nlohmann::ordered_json;
    class GridExperiment : public GridBase {
    public:
        explicit GridExperiment(struct ConfigGrid& config);
        void go(struct ConfigMPI& config_mpi);
        ~GridExperiment() = default;
        json loadResults();
    private:
        void save(json& results);
        json initializeResults();
        json build_tasks_mpi();
    };
    /* *************************************************************************************************************
    //
    // MPI Search Functions
    //
    ************************************************************************************************************* */
    class MPI_EXPERIMENT {
    public:
        static std::string get_color_rank(int rank)
        {
            auto colors = { Colors::WHITE(), Colors::RED(), Colors::GREEN(),  Colors::BLUE(), Colors::MAGENTA(), Colors::CYAN(), Colors::YELLOW(), Colors::BLACK() };
            std::string id = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
            auto idx = rank % id.size();
            return *(colors.begin() + rank % colors.size()) + id[idx];
        }
        static json producer(std::vector<std::string>& names, json& tasks, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result)
        {
            Task_Result result;
            json results;
            int num_tasks = tasks.size();

            //
            // 2a.1 Producer will loop to send all the tasks to the consumers and receive the results
            //
            for (int i = 0; i < num_tasks; ++i) {
                MPI_Status status;
                MPI_Recv(&result, 1, MPI_Result, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == TAG_RESULT) {
                    //Store result
                    store_search_result(names, result, results);
                }
                MPI_Send(&i, 1, MPI_INT, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
            }
            //
            // 2a.2 Producer will send the end message to all the consumers
            //
            for (int i = 0; i < config_mpi.n_procs - 1; ++i) {
                MPI_Status status;
                MPI_Recv(&result, 1, MPI_Result, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == TAG_RESULT) {
                    //Store result
                    store_search_result(names, result, results);
                }
                MPI_Send(&i, 1, MPI_INT, status.MPI_SOURCE, TAG_END, MPI_COMM_WORLD);
            }
            return results;
        }
        static void consumer(Datasets& datasets, json& tasks, struct ConfigGrid& config, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result)
        {
            Task_Result result;
            //
            // 2b.1 Consumers announce to the producer that they are ready to receive a task
            //
            MPI_Send(&result, 1, MPI_Result, config_mpi.manager, TAG_QUERY, MPI_COMM_WORLD);
            int task;
            while (true) {
                MPI_Status status;
                //
                // 2b.2 Consumers receive the task from the producer and process it
                //
                MPI_Recv(&task, 1, MPI_INT, config_mpi.manager, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == TAG_END) {
                    break;
                }
                mpi_experiment_consumer_go(config, config_mpi, tasks, task, datasets, &result);
                //
                // 2b.3 Consumers send the result to the producer
                //
                MPI_Send(&result, 1, MPI_Result, config_mpi.manager, TAG_RESULT, MPI_COMM_WORLD);
            }
        }
        static void select_best_results_folds(json& results, json& all_results, std::string& model)
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
        static json store_search_result(std::vector<std::string>& names, Task_Result& result, json& results)
        {
            json json_result = {
                { "score", result.score },
                { "combination", result.idx_combination },
                { "fold", result.n_fold },
                { "time", result.time },
                { "dataset", result.idx_dataset }
            };
            auto name = names[result.idx_dataset];
            if (!results.contains(name)) {
                results[name] = json::array();
            }
            results[name].push_back(json_result);
            return results;
        }
        static void consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result)
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
            //
            // Update progress bar
            //
            std::cout << get_color_rank(config_mpi.rank) << std::flush;
        }
    };
} /* namespace platform */
#endif