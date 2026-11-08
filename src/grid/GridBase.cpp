#include <sstream>
#include <algorithm>
#include <random>
#include <cstddef>
#include <cstdio>
#include "common/DotEnv.h"
#include "common/Paths.h"
#include "common/Colors.h"
#include "GridBase.h"


namespace platform {

    GridBase::GridBase(struct ConfigGrid& config)
    {
        this->config = config;
        auto env = platform::DotEnv();
        this->config.platform = env.get("platform");

    }
    void GridBase::validate_config()
    {
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
    }
    void GridBase::shuffle_and_progress_bar(json& tasks)
    {
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
    }
    json GridBase::build_tasks(Datasets& datasets)
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
                    };
                    tasks.push_back(task);
                }
            }
        }
        shuffle_and_progress_bar(tasks);
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
        std::cout << Colors::MAGENTA() << " W  " << std::setw(max_dataset) << std::left << "Dataset";
        std::cout << " Seed Fold Time" << std::endl;
        std::cout << "=== " << std::string(max_dataset, '=') << " ==== ==== " << std::string(15, '=') << std::endl;
        for (int worker = 0; worker < config_mpi.n_procs; ++worker) {
            auto color = (worker % 2) ? Colors::CYAN() : Colors::BLUE();
            std::cout << color << std::right << std::setw(3) << worker << " ";
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
                std::cout << std::left << std::setw(max_dataset) << task["dataset"].get<std::string>();
                std::cout << " " << std::setw(4) << std::right << task["seed"].get<int>();
                std::cout << " " << std::setw(4) << task["fold"].get<int>();
                std::cout << " " << std::setw(15) << std::setprecision(7) << std::fixed << task["time"].get<double>() << std::endl;
                total += task["time"].get<double>();
            }
            if (num_tasks > 1) {
                std::cout << Colors::MAGENTA() << "    ";
                std::cout << std::setw(max_dataset) << "Total (" << std::setw(2) << std::right << num_tasks << ")" << std::string(7, '.');
                std::cout << " " << std::setw(15) << std::setprecision(7) << std::fixed << total << std::endl;
            }
        }
    }
    void GridBase::go(struct ConfigMPI& config_mpi)
    {
        /*
        * Each task is a json object with the data needed by the process
        *
        * The overall process consists in these steps:
           * 0. Validate config, create the MPI result type & tasks
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
           * 3. Manager compile results for each dataset
           * 3.1 Loop thru all the results obtained from each outer fold (task) and select the best
           * 3.2 Save the results
           * 3.3 Summary of jobs done
        */
        //
        // 0.1 Create the MPI result type
        //
        validate_config();
        Task_Result result;
        int tasks_size;
        MPI_Datatype MPI_Result;
        const int n_fields = 17;
        MPI_Datatype type[n_fields] = { MPI_UNSIGNED, MPI_UNSIGNED, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT,
            MPI_INT, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT, MPI_INT };
        int blocklen[n_fields];
        std::fill(blocklen, blocklen + n_fields, 1);
        MPI_Aint disp[n_fields];
        disp[0] = offsetof(Task_Result, idx_dataset);
        disp[1] = offsetof(Task_Result, idx_combination);
        disp[2] = offsetof(Task_Result, n_fold);
        disp[3] = offsetof(Task_Result, score);
        disp[4] = offsetof(Task_Result, time);
        disp[5] = offsetof(Task_Result, time_train);
        disp[6] = offsetof(Task_Result, nodes);
        disp[7] = offsetof(Task_Result, leaves);
        disp[8] = offsetof(Task_Result, depth);
        disp[9] = offsetof(Task_Result, process);
        disp[10] = offsetof(Task_Result, task);
        disp[11] = offsetof(Task_Result, seed);
        disp[12] = offsetof(Task_Result, score_train);
        disp[13] = offsetof(Task_Result, has_train);
        disp[14] = offsetof(Task_Result, n_classes);
        disp[15] = offsetof(Task_Result, notes_len);
        disp[16] = offsetof(Task_Result, graph_len);
        MPI_Datatype tmp_type;
        MPI_Type_create_struct(n_fields, blocklen, disp, type, &tmp_type);
        // Resize to sizeof(Task_Result) so trailing padding is covered (arrays of results, if ever)
        MPI_Type_create_resized(tmp_type, 0, sizeof(Task_Result), &MPI_Result);
        MPI_Type_free(&tmp_type);
        MPI_Type_commit(&MPI_Result);
        //
        // 0.2 Manager creates the tasks
        //
        char* msg;
        json tasks;
        auto env = platform::DotEnv();
        // El algoritmo de discretización sale de --discretize-algo cuando se ha
        // indicado; si no, del .env. Antes se leía siempre del .env, así que un
        // --discretize-algo distinto acababa en el informe pero no se aplicaba.
        auto discretize_algo = config.discretize_algo.empty() ? env.get("discretize_algo") : config.discretize_algo;
        auto datasets = Datasets(config.discretize, Paths::datasets(), discretize_algo);
        if (config_mpi.rank == config_mpi.manager) {
            timer.start();
            tasks = build_tasks(datasets);
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


        if (config_mpi.rank == config_mpi.manager) {
            //
            // 2a. Producer delivers the tasks to the consumers
            //
            auto datasets_names = filterDatasets(datasets);
            json all_results = producer(datasets_names, tasks, config_mpi, MPI_Result);
            std::cout << separator << std::endl;
            //
            // 3. Manager compile results for each dataset
            //
            auto results = initializeResults();
            compile_results(results, all_results, config.model);
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
    json GridBase::producer(std::vector<std::string>& names, json& tasks, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result)
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
                // The worker sends the rest of the result right after it; MPI keeps the
                // order of the messages between two processes, so they are the next ones
                // from status.MPI_SOURCE.
                Task_Payload payload;
                receive_payload(result, payload, status.MPI_SOURCE);
                //Store result
                store_result(names, result, payload, results);
                // Display progress in the manager process using the worker's rank
                std::cout << get_color_rank(result.process) << std::flush;
                std::cout.flush();
                std::fflush(stdout);
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
                // The worker sends the rest of the result right after it; MPI keeps the
                // order of the messages between two processes, so they are the next ones
                // from status.MPI_SOURCE.
                Task_Payload payload;
                receive_payload(result, payload, status.MPI_SOURCE);
                //Store result
                store_result(names, result, payload, results);
                // Display progress in the manager process using the worker's rank
                std::cout << get_color_rank(result.process) << std::flush;
                std::cout.flush();
                std::fflush(stdout);
            }
            MPI_Send(&i, 1, MPI_INT, status.MPI_SOURCE, TAG_END, MPI_COMM_WORLD);
        }
        return results;
    }
    void GridBase::abort_task(struct ConfigMPI& config_mpi, json& tasks, int n_task, const std::string& message)
    {
        const auto& task = tasks[n_task];
        // One write, so the messages of two failing ranks do not interleave
        std::ostringstream oss;
        oss << "b_grid: task " << n_task << " failed on rank " << config_mpi.rank;
        for (const auto& key : { "dataset", "seed", "fold" }) {
            if (task.contains(key))
                oss << " " << key << "=" << task[key].dump();
        }
        oss << ": " << message << "\n";
        std::cerr << oss.str() << std::flush;
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort(); // MPI_Abort is not declared noreturn
    }
    void GridBase::send_payload(const Task_Result& result, const Task_Payload& payload, int manager)
    {
        int cells = result.n_classes * result.n_classes;
        if (result.n_classes > 0) {
            MPI_Send(payload.cm_test.data(), cells, MPI_INT, manager, TAG_CONFUSION, MPI_COMM_WORLD);
            if (result.has_train)
                MPI_Send(payload.cm_train.data(), cells, MPI_INT, manager, TAG_CONFUSION_TRAIN, MPI_COMM_WORLD);
        }
        if (result.notes_len > 0)
            MPI_Send(payload.notes.data(), result.notes_len, MPI_CHAR, manager, TAG_NOTES, MPI_COMM_WORLD);
        if (result.graph_len > 0)
            MPI_Send(payload.graph.data(), result.graph_len, MPI_CHAR, manager, TAG_GRAPH, MPI_COMM_WORLD);
    }
    void GridBase::receive_payload(const Task_Result& result, Task_Payload& payload, int source)
    {
        MPI_Status status;
        int cells = result.n_classes * result.n_classes;
        if (result.n_classes > 0) {
            payload.cm_test.resize(cells);
            MPI_Recv(payload.cm_test.data(), cells, MPI_INT, source, TAG_CONFUSION, MPI_COMM_WORLD, &status);
            if (result.has_train) {
                payload.cm_train.resize(cells);
                MPI_Recv(payload.cm_train.data(), cells, MPI_INT, source, TAG_CONFUSION_TRAIN, MPI_COMM_WORLD, &status);
            }
        }
        if (result.notes_len > 0) {
            payload.notes.resize(result.notes_len);
            MPI_Recv(payload.notes.data(), result.notes_len, MPI_CHAR, source, TAG_NOTES, MPI_COMM_WORLD, &status);
        }
        if (result.graph_len > 0) {
            payload.graph.resize(result.graph_len);
            MPI_Recv(payload.graph.data(), result.graph_len, MPI_CHAR, source, TAG_GRAPH, MPI_COMM_WORLD, &status);
        }
    }
    void GridBase::consumer(Datasets& datasets, json& tasks, struct ConfigGrid& config, struct ConfigMPI& config_mpi, MPI_Datatype& MPI_Result)
    {
        Task_Result result{};
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
            Task_Payload payload;
            try {
                consumer_go(config, config_mpi, tasks, task, datasets, &result, payload);
            }
            catch (const std::exception& e) {
                // A failed task must stop the whole run, as in b_main, and say where it
                // failed. Leaving the loop would let the manager wait forever.
                abort_task(config_mpi, tasks, task, e.what());
            }
            catch (...) {
                abort_task(config_mpi, tasks, task, "unknown exception");
            }
            //
            // 2b.3 Consumers send the result to the producer
            //
            MPI_Send(&result, 1, MPI_Result, config_mpi.manager, TAG_RESULT, MPI_COMM_WORLD);
            send_payload(result, payload, config_mpi.manager);
        }
    }

}
