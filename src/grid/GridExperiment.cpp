#include <iostream>
#include <cstddef>
#include <torch/torch.h>
#include <folding.hpp>
#include "main/Models.h"
#include "common/Paths.h"
#include "common/Utils.h"
#include "GridExperiment.h"

namespace platform {
    GridExperiment::GridExperiment(argparse::ArgumentParser& program, struct ConfigGrid& config) : arguments(program), GridBase(config)
    {
        std::string file_name, model_name, title, hyperparameters_file, datasets_file, discretize_algo, smooth_strat, score;
        json hyperparameters_json;
        bool discretize_dataset, stratified, hyper_best;
        std::vector<int> seeds;
        std::vector<std::string> file_names;
        int n_folds;
        file_name = program.get<std::string>("dataset");
        file_names = program.get<std::vector<std::string>>("datasets");
        datasets_file = program.get<std::string>("datasets-file");
        model_name = program.get<std::string>("model");
        discretize_dataset = program.get<bool>("discretize");
        discretize_algo = program.get<std::string>("discretize-algo");
        smooth_strat = program.get<std::string>("smooth-strat");
        stratified = program.get<bool>("stratified");
        n_folds = program.get<int>("folds");
        score = program.get<std::string>("score");
        seeds = program.get<std::vector<int>>("seeds");
        auto hyperparameters = program.get<std::string>("hyperparameters");
        hyperparameters_json = json::parse(hyperparameters);
        hyperparameters_file = program.get<std::string>("hyper-file");
        hyper_best = program.get<bool>("hyper-best");
        if (hyper_best) {
            // Build the best results file_name
            hyperparameters_file = platform::Paths::results() + platform::Paths::bestResultsFile(score, model_name);
            // ignore this parameter
            hyperparameters = "{}";
        } else {
            if (hyperparameters_file != "" && hyperparameters != "{}") {
                throw runtime_error("hyperparameters and hyper_file are mutually exclusive");
            }
        }
        title = program.get<std::string>("title");
        if (title == "" && file_name == "all") {
            throw runtime_error("title is mandatory if all datasets are to be tested");
        }
        auto datasets = platform::Datasets(false, platform::Paths::datasets());
        if (datasets_file != "") {
            ifstream catalog(datasets_file);
            if (catalog.is_open()) {
                std::string line;
                while (getline(catalog, line)) {
                    if (line.empty() || line[0] == '#') {
                        continue;
                    }
                    if (!datasets.isDataset(line)) {
                        cerr << "Dataset " << line << " not found" << std::endl;
                        exit(1);
                    }
                    filesToTest.push_back(line);
                }
                catalog.close();
                if (title == "") {
                    title = "Test " + to_string(filesToTest.size()) + " datasets (" + datasets_file + ") "\
                        + model_name + " " + to_string(n_folds) + " folds";
                }
            } else {
                throw std::invalid_argument("Unable to open catalog file. [" + datasets_file + "]");
            }
        } else {
            if (file_names.size() > 0) {
                for (auto file : file_names) {
                    if (!datasets.isDataset(file)) {
                        cerr << "Dataset " << file << " not found" << std::endl;
                        exit(1);
                    }
                }
                filesToTest = file_names;
                if (title == "") {
                    title = "Test " + to_string(file_names.size()) + " datasets " + model_name + " " + to_string(n_folds) + " folds";
                }
            } else {
                if (file_name != "all") {
                    if (!datasets.isDataset(file_name)) {
                        cerr << "Dataset " << file_name << " not found" << std::endl;
                        exit(1);
                    }
                    if (title == "") {
                        title = "Test " + file_name + " " + model_name + " " + to_string(n_folds) + " folds";
                    }
                    filesToTest.push_back(file_name);
                } else {
                    filesToTest = datasets.getNames();
                }
            }
        }

        platform::HyperParameters test_hyperparams;
        if (hyperparameters_file != "") {
            test_hyperparams = platform::HyperParameters(datasets.getNames(), hyperparameters_file, hyper_best);
        } else {
            test_hyperparams = platform::HyperParameters(datasets.getNames(), hyperparameters_json);
        }
        this->config.model = model_name;
        this->config.score = score;
        this->config.discretize = discretize_dataset;
        this->config.stratified = stratified;
        this->config.smooth_strategy = smooth_strat;
        this->config.n_folds = n_folds;
        this->config.seeds = seeds;
        auto env = platform::DotEnv();
        experiment.setTitle(title).setLanguage("c++").setLanguageVersion("gcc 14.1.1");
        experiment.setDiscretizationAlgorithm(discretize_algo).setSmoothSrategy(smooth_strat);
        experiment.setDiscretized(discretize_dataset).setModel(model_name).setPlatform(env.get("platform"));
        experiment.setStratified(stratified).setNFolds(n_folds).setScoreName(score);
        experiment.setHyperparameters(test_hyperparams);
        for (auto seed : seeds) {
            experiment.addRandomSeed(seed);
        }
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
        * }
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
    std::vector<std::string> GridExperiment::filterDatasets(Datasets& datasets) const
    {
        // Load datasets
        // auto datasets_names = datasets.getNames();
        // datasets_names.clear();
        // datasets_names.push_back("iris");
        // datasets_names.push_back("wine");
        // datasets_names.push_back("balance-scale");
        return filesToTest;
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
        results = json::array();
        auto datasets = Datasets(false, Paths::datasets());
        for (const auto& result_item : all_results.items()) {
            // each result has the results of all the outer folds as each one were a different task
            auto dataset_name = result_item.key();
            auto data = result_item.value();
            auto result = json::object();
            int data_size = data.size();
            auto score = torch::zeros({ data_size }, torch::kFloat64);
            auto time_t = torch::zeros({ data_size }, torch::kFloat64);
            auto nodes = torch::zeros({ data_size }, torch::kFloat64);
            auto leaves = torch::zeros({ data_size }, torch::kFloat64);
            auto depth = torch::zeros({ data_size }, torch::kFloat64);
            for (int fold = 0; fold < data_size; ++fold) {
                result["scores_test"].push_back(data[fold]["score"]);
                score[fold] = data[fold]["score"].get<double>();
                time_t[fold] = data[fold]["time"].get<double>();
                nodes[fold] = data[fold]["nodes"].get<double>();
                leaves[fold] = data[fold]["leaves"].get<double>();
                depth[fold] = data[fold]["depth"].get<double>();
            }
            double score_mean = torch::mean(score).item<double>();
            double score_std = torch::std(score).item<double>();
            double time_mean = torch::mean(time_t).item<double>();
            double time_std = torch::std(time_t).item<double>();
            double nodes_mean = torch::mean(nodes).item<double>();
            double leaves_mean = torch::mean(leaves).item<double>();
            double depth_mean = torch::mean(depth).item<double>();
            auto& dataset = datasets.getDataset(dataset_name);
            dataset.load();
            result["samples"] = dataset.getNSamples();
            result["features"] = dataset.getNFeatures();
            result["classes"] = dataset.getNClasses();
            result["hyperparameters"] = experiment.getHyperParameters().get(dataset_name);
            result["score"] = score_mean;
            result["score_std"] = score_std;
            result["time"] = time_mean;
            result["time_std"] = time_std;
            result["nodes"] = nodes_mean;
            result["leaves"] = leaves_mean;
            result["depth"] = depth_mean;
            result["dataset"] = dataset_name;
            // Fixed data
            result["scores_train"] = json::array();
            result["times_train"] = json::array();
            result["times_test"] = json::array();
            result["train_time"] = 0.0;
            result["train_time_std"] = 0.0;
            result["test_time"] = 0.0;
            result["test_time_std"] = 0.0;
            result["score_train"] = 0.0;
            result["score_train_std"] = 0.0;
            result["confusion_matrices"] = json::array();
            results.push_back(result);
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
        auto hyperparameters = experiment.getHyperParameters();
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