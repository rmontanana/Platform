#include <iostream>
#include <cstddef>
#include <torch/torch.h>
#include <folding.hpp>
#include "main/Models.h"
#include "common/Paths.h"
#include "common/Colors.h"
#include "common/Utils.h"
#include "GridSearch.h"

namespace platform {

    std::string get_color_rank(int rank)
    {
        auto colors = { Colors::WHITE(), Colors::RED(), Colors::GREEN(),  Colors::BLUE(), Colors::MAGENTA(), Colors::CYAN() };
        return *(colors.begin() + rank % colors.size());
    }
    GridSearch::GridSearch(struct ConfigGrid& config) : config(config)
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
    json GridSearch::build_tasks_mpi(int rank)
    {
        auto tasks = json::array();
        auto grid = GridData(Paths::grid_input(config.model));
        auto datasets = Datasets(false, Paths::datasets());
        auto all_datasets = datasets.getNames();
        auto datasets_names = filterDatasets(datasets);
        for (int idx_dataset = 0; idx_dataset < datasets_names.size(); ++idx_dataset) {
            auto dataset = datasets_names[idx_dataset];
            for (const auto& seed : config.seeds) {
                auto combinations = grid.getGrid(dataset);
                for (int n_fold = 0; n_fold < config.n_folds; n_fold++) {
                    json task = {
                        { "dataset", dataset },
                        { "idx_dataset", idx_dataset},
                        { "seed", seed },
                        { "fold", n_fold},
                    };
                    tasks.push_back(task);
                }
            }
        }
        // Shuffle the array so heavy datasets are spread across the workers
        std::mt19937 g{ 271 }; // Use fixed seed to obtain the same shuffle
        std::shuffle(tasks.begin(), tasks.end(), g);
        std::cout << get_color_rank(rank) << "* Number of tasks: " << tasks.size() << std::endl;
        std::cout << separator;
        for (int i = 0; i < tasks.size(); ++i) {
            std::cout << (i + 1) % 10;
        }
        std::cout << separator << std::endl << separator << std::flush;
        return tasks;
    }
    void process_task_mpi_consumer(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result)
    {
        // initialize
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
        // Generate the hyperparamters combinations
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
        double best_fold_score = 0.0;
        int best_idx_combination = -1;
        bayesnet::Smoothing_t smoothing = bayesnet::Smoothing_t::NONE;
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
                // Nested level fold
                auto [train_nested, test_nested] = nested_fold->getFold(n_nested_fold);
                auto train_nested_t = torch::tensor(train_nested);
                auto test_nested_t = torch::tensor(test_nested);
                auto X_nested_train = X_train.index({ "...", train_nested_t });
                auto y_nested_train = y_train.index({ train_nested_t });
                auto X_nested_test = X_train.index({ "...", test_nested_t });
                auto y_nested_test = y_train.index({ test_nested_t });
                // Build Classifier with selected hyperparameters
                auto clf = Models::instance()->create(config.model);
                auto valid = clf->getValidHyperparameters();
                hyperparameters.check(valid, dataset_name);
                clf->setHyperparameters(hyperparameters.get(dataset_name));
                // Train model
                clf->fit(X_nested_train, y_nested_train, features, className, states, smoothing);
                // Test model
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
        // Build Classifier with the best hyperparameters to obtain the best score
        auto hyperparameters = platform::HyperParameters(datasets.getNames(), best_fold_hyper);
        auto clf = Models::instance()->create(config.model);
        auto valid = clf->getValidHyperparameters();
        hyperparameters.check(valid, dataset_name);
        clf->setHyperparameters(best_fold_hyper);
        clf->fit(X_train, y_train, features, className, states, smoothing);
        best_fold_score = clf->score(X_test, y_test);
        // Return the result
        result->idx_dataset = task["idx_dataset"].get<int>();
        result->idx_combination = best_idx_combination;
        result->score = best_fold_score;
        result->n_fold = n_fold;
        result->time = timer.getDuration();
        // Update progress bar
        std::cout << get_color_rank(config_mpi.rank) << "*" << std::flush;
    }
    json store_result(std::vector<std::string>& names, Task_Result& result, json& results)
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
    json producer(std::vector<std::string>& names, json& tasks, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result)
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
                store_result(names, result, results);
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
                store_result(names, result, results);
            }
            MPI_Send(&i, 1, MPI_INT, status.MPI_SOURCE, TAG_END, MPI_COMM_WORLD);
        }
        return results;
    }
    void select_best_results_folds(json& results, json& all_results, std::string& model)
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
    void consumer(Datasets& datasets, json& tasks, struct ConfigGrid& config, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result)
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
            process_task_mpi_consumer(config, config_mpi, tasks, task, datasets, &result);
            //
            // 2b.3 Consumers send the result to the producer
            //
            MPI_Send(&result, 1, MPI_Result, config_mpi.manager, TAG_RESULT, MPI_COMM_WORLD);
        }
    }
    void GridSearch::go(struct ConfigMPI& config_mpi)
    {
        /*
        * Each task is a json object with the following structure:
        * {
        *   "dataset": "dataset_name",
        *   "idx_dataset": idx_dataset, // used to identify the dataset in the results
        *    // this index is relative to the used datasets in the actual run not to the whole datasets
        *   "seed": # of seed to use,
        *   "Fold": # of fold to process
        * }
        *
        * The overall process consists in these steps:
           * 0. Create the MPI result type & tasks
           * 0.1 Create the MPI result type
           * 0.2 Manager creates the tasks
           * 1. Manager will broadcast the tasks to all the processes
           * 1.1 Broadcast the number of tasks
           * 1.2 Broadcast the length of the following string
           * 1.2 Broadcast the tasks as a char* string
           * 2a. Producer delivers the tasks to the consumers
           * 2a.1 Producer will loop to send all the tasks to the consumers and receive the results
           * 2a.2 Producer will send the end message to all the consumers
           * 2b. Consumers process the tasks and send the results to the producer
           * 2b.1 Consumers announce to the producer that they are ready to receive a task
           * 2b.2 Consumers receive the task from the producer and process it
           * 2b.3 Consumers send the result to the producer
           * 3. Manager select the bests sccores for each dataset
           * 3.1 Loop thru all the results obtained from each outer fold (task) and select the best
           * 3.2 Save the results
        */
        //
        // 0.1 Create the MPI result type
        //
        Task_Result result;
        int tasks_size;
        MPI_Datatype MPI_Result;
        MPI_Datatype type[5] = { MPI_UNSIGNED, MPI_UNSIGNED, MPI_INT, MPI_DOUBLE, MPI_DOUBLE };
        int blocklen[5] = { 1, 1, 1, 1, 1 };
        MPI_Aint disp[5];
        disp[0] = offsetof(Task_Result, idx_dataset);
        disp[1] = offsetof(Task_Result, idx_combination);
        disp[2] = offsetof(Task_Result, n_fold);
        disp[3] = offsetof(Task_Result, score);
        disp[4] = offsetof(Task_Result, time);
        MPI_Type_create_struct(5, blocklen, disp, type, &MPI_Result);
        MPI_Type_commit(&MPI_Result);
        //
        // 0.2 Manager creates the tasks
        //
        char* msg;
        json tasks;
        if (config_mpi.rank == config_mpi.manager) {
            timer.start();
            tasks = build_tasks_mpi(config_mpi.rank);
            auto tasks_str = tasks.dump();
            tasks_size = tasks_str.size();
            msg = new char[tasks_size + 1];
            strcpy(msg, tasks_str.c_str());
        }
        //
        // 1. Manager will broadcast the tasks to all the processes
        //
        MPI_Bcast(&tasks_size, 1, MPI_INT, config_mpi.manager, MPI_COMM_WORLD);
        if (config_mpi.rank != config_mpi.manager) {
            msg = new char[tasks_size + 1];
        }
        MPI_Bcast(msg, tasks_size + 1, MPI_CHAR, config_mpi.manager, MPI_COMM_WORLD);
        tasks = json::parse(msg);
        delete[] msg;
        auto env = platform::DotEnv();
        auto datasets = Datasets(config.discretize, Paths::datasets(), env.get("discretize_algo"));

        if (config_mpi.rank == config_mpi.manager) {
            //
            // 2a. Producer delivers the tasks to the consumers
            //
            auto datasets_names = filterDatasets(datasets);
            json all_results = producer(datasets_names, tasks, config_mpi, MPI_Result);
            std::cout << get_color_rank(config_mpi.rank) << separator << std::endl;
            //
            // 3. Manager select the bests sccores for each dataset
            //
            auto results = initializeResults();
            select_best_results_folds(results, all_results, config.model);
            //
            // 3.2 Save the results
            //
            save(results);
        } else {
            //
            // 2b. Consumers process the tasks and send the results to the producer
            //
            consumer(datasets, tasks, config, config_mpi, MPI_Result);
        }
    }
    json GridSearch::initializeResults()
    {
        // Load previous results if continue is set
        json results;
        if (config.continue_from != NO_CONTINUE()) {
            if (!config.quiet)
                std::cout << "* Loading previous results" << std::endl;
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
} /* namespace platform */