#include <random>
#include <cstddef>
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
    std::string GridBase::get_color_rank(int rank)
    {
        auto colors = { Colors::WHITE(), Colors::RED(), Colors::GREEN(),  Colors::BLUE(), Colors::MAGENTA(), Colors::CYAN(), Colors::YELLOW(), Colors::BLACK() };
        std::string id = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        auto idx = rank % id.size();
        return *(colors.begin() + rank % colors.size()) + id[idx];
    };
    json GridBase::build_tasks()
    {
        /*
        * Each task is a json object with the following structure:
        * {
        *   "dataset": "dataset_name",
        *   "idx_dataset": idx_dataset, // used to identify the dataset in the results
        *    // this index is relative to the list of used datasets in the actual run not to the whole datasets list
        *   "seed": # of seed to use,
        *   "fold": # of fold to process
        * }
        */
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
        // Shuffle the array so heavy datasets are eas  ier spread across the workers
        std::mt19937 g{ 271 }; // Use fixed seed to obtain the same shuffle
        std::shuffle(tasks.begin(), tasks.end(), g);
        std::cout << "* Number of tasks: " << tasks.size() << std::endl;
        std::cout << separator << std::flush;
        for (int i = 0; i < tasks.size(); ++i) {
            if ((i + 1) % 10 == 0)
                std::cout << separator;
            else
                std::cout << (i + 1) % 10;
        }
        std::cout << separator << std::endl << separator << std::flush;
        return tasks;
    }
    void GridBase::summary(json& all_results, json& tasks, struct ConfigMPI& config_mpi)
    {
        // Report the tasks done by each worker, showing dataset number, seed, fold and time spent
        // The format I want to show is:
        // worker, dataset, seed, fold, time
        // with headers
        std::cout << Colors::RESET() << "* Summary of tasks done by each worker" << std::endl;
        json worker_tasks = json::array();
        for (int i = 0; i < config_mpi.n_procs; ++i) {
            worker_tasks.push_back(json::array());
        }
        int max_dataset = 7;
        for (const auto& [key, results] : all_results.items()) {
            auto dataset = key;
            if (dataset.size() > max_dataset)
                max_dataset = dataset.size();
            for (const auto& result : results) {
                int n_task = result["task"].get<int>();
                json task = tasks[n_task];
                auto seed = task["seed"].get<int>();
                auto fold = task["fold"].get<int>();
                auto time = result["time"].get<double>();
                auto worker = result["process"].get<int>();
                json line = {
                    { "dataset", dataset },
                    { "seed", seed },
                    { "fold", fold },
                    { "time", time }
                };
                worker_tasks[worker].push_back(line);
            }
        }
        std::cout << Colors::MAGENTA() << " W  " << setw(max_dataset) << std::left << "Dataset";
        std::cout << " Seed Fold Time" << std::endl;
        std::cout << "=== " << std::string(max_dataset, '=') << " ==== ==== " << std::string(15, '=') << std::endl;
        for (int worker = 0; worker < config_mpi.n_procs; ++worker) {
            auto color = (worker % 2) ? Colors::CYAN() : Colors::BLUE();
            std::cout << color << std::right << setw(3) << worker << " ";
            if (worker == config_mpi.manager) {
                std::cout << "Manager" << std::endl;
                continue;
            }
            if (worker_tasks[worker].empty()) {
                std::cout << "No tasks" << std::endl;
                continue;
            }
            bool first = true;
            double total = 0.0;
            int num_tasks = 0;
            for (const auto& task : worker_tasks[worker]) {
                num_tasks++;
                if (!first)
                    std::cout << std::string(4, ' ');
                else
                    first = false;
                std::cout << std::left << setw(max_dataset) << task["dataset"].get<std::string>();
                std::cout << " " << setw(4) << std::right << task["seed"].get<int>();
                std::cout << " " << setw(4) << task["fold"].get<int>();
                std::cout << " " << setw(15) << std::setprecision(7) << std::fixed << task["time"].get<double>() << std::endl;
                total += task["time"].get<double>();
            }
            if (num_tasks > 1) {
                std::cout << Colors::MAGENTA() << setw(3) << std::right << num_tasks;
                std::cout << setw(max_dataset) << " Total..." << std::string(10, '.');
                std::cout << setw(15) << std::setprecision(7) << std::fixed << total << std::endl;
            }
        }
    }
    void GridBase::go(struct ConfigMPI& config_mpi)
    {
        /*
        * Each task is a json object with the following structure:
        * {
        *   "dataset": "dataset_name",
        *   "idx_dataset": idx_dataset, // used to identify the dataset in the results
        *    // this index is relative to the list of used datasets in the actual run not to the whole datasets list
        *   "seed": # of seed to use,
        *   "fold": # of fold to process
        * }
        *
        * This way a task consists in process all combinations of hyperparameters for a dataset, seed and fold
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
           * 3. Manager select the bests scores for each dataset
           * 3.1 Loop thru all the results obtained from each outer fold (task) and select the best
           * 3.2 Save the results
           * 3.3 Summary of jobs done
        */
        //
        // 0.1 Create the MPI result type
        //
        Task_Result result;
        int tasks_size;
        MPI_Datatype MPI_Result;
        MPI_Datatype type[10] = { MPI_UNSIGNED, MPI_UNSIGNED, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT };
        int blocklen[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        MPI_Aint disp[10];
        disp[0] = offsetof(Task_Result, idx_dataset);
        disp[1] = offsetof(Task_Result, idx_combination);
        disp[2] = offsetof(Task_Result, n_fold);
        disp[3] = offsetof(Task_Result, score);
        disp[4] = offsetof(Task_Result, time);
        disp[5] = offsetof(Task_Result, nodes);
        disp[6] = offsetof(Task_Result, leaves);
        disp[7] = offsetof(Task_Result, depth);
        disp[8] = offsetof(Task_Result, process);
        disp[9] = offsetof(Task_Result, task);
        MPI_Type_create_struct(10, blocklen, disp, type, &MPI_Result);
        MPI_Type_commit(&MPI_Result);
        //
        // 0.2 Manager creates the tasks
        //
        char* msg;
        json tasks;
        if (config_mpi.rank == config_mpi.manager) {
            timer.start();
            tasks = build_tasks();
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
            std::cout << separator << std::endl;
            //
            // 3. Manager select the bests sccores for each dataset
            //
            auto results = initializeResults();
            select_best_results_folds(results, all_results, config.model);
            //
            // 3.2 Save the results
            //
            save(results);
            //
            // 3.3 Summary of jobs done
            //
            if (!config.quiet)
                summary(all_results, tasks, config_mpi);
        } else {
            //
            // 2b. Consumers process the tasks and send the results to the producer
            //
            consumer(datasets, tasks, config, config_mpi, MPI_Result);
        }
    }

}