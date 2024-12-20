#include <iostream>
#include <cstddef>
#include <torch/torch.h>
#include <folding.hpp>
#include "main/Models.h"
#include "common/Paths.h"
#include "common/Colors.h"
#include "common/Utils.h"
#include "GridExperiment.h"

namespace platform {

    GridExperiment::GridExperiment(struct ConfigGrid& config) : GridBase(config)
    {
    }
    json GridExperiment::loadResults()
    {
        std::ifstream file(Paths::grid_output(config.model));
        if (file.is_open()) {
            return json::parse(file);
        }
        return json();
    }
    json GridExperiment::build_tasks_mpi()
    {
        auto tasks = json::array();
        auto grid = GridData(Paths::grid_input(config.model));
        auto datasets = Datasets(false, Paths::datasets());
        auto all_datasets = datasets.getNames();
        auto datasets_names = all_datasets;
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
    void GridExperiment::go(struct ConfigMPI& config_mpi)
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
            tasks = build_tasks_mpi();
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
            auto datasets_names = std::vector<std::string>();
            json all_results = mpi_experiment_producer(datasets_names, tasks, config_mpi, MPI_Result);
            std::cout << separator << std::endl;
            //
            // 3. Manager select the bests sccores for each dataset
            //
            auto results = initializeResults();
            //select_best_results_folds(results, all_results, config.model);
            //
            // 3.2 Save the results
            //
            save(results);
        } else {
            //
            // 2b. Consumers prostore_search_resultcess the tasks and send the results to the producer
            //
            mpi_experiment_consumer(datasets, tasks, config, config_mpi, MPI_Result);
        }
    }
    json GridExperiment::initializeResults()
    {
        // Load previous results if continue is set
        json results;
        return results;
    }
    void GridExperiment::save(json& results)
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