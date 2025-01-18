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
        saveResults = program.get<bool>("save");
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
                saveResults = true;
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
                saveResults = true;
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
                    saveResults = true;
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
        this->config.quiet = false;
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
    std::vector<std::string> GridExperiment::filterDatasets(Datasets& datasets) const
    {
        return filesToTest;
    }
    json GridExperiment::initializeResults()
    {
        json results;
        return results;
    }
    void GridExperiment::save(json& results)
    {
    }
    void GridExperiment::compile_results(json& results, json& all_results, std::string& model)
    {
        auto datasets = Datasets(false, Paths::datasets());
        nlohmann::json temp = all_results; // To restore the order of the data by dataset name
        all_results = temp;
        for (const auto& result_item : all_results.items()) {
            // each result has the results of all the outer folds as each one were a different task
            auto dataset_name = result_item.key();
            auto data = result_item.value();
            auto result = json::object();
            int data_size = data.size();
            auto score = torch::zeros({ data_size }, torch::kFloat64);
            auto score_train = torch::zeros({ data_size }, torch::kFloat64);
            auto time_test = torch::zeros({ data_size }, torch::kFloat64);
            auto time_train = torch::zeros({ data_size }, torch::kFloat64);
            auto nodes = torch::zeros({ data_size }, torch::kFloat64);
            auto leaves = torch::zeros({ data_size }, torch::kFloat64);
            auto depth = torch::zeros({ data_size }, torch::kFloat64);
            auto& dataset = datasets.getDataset(dataset_name);
            dataset.load();
            //
            // Prepare Result
            //
            auto partial_result = PartialResult();
            partial_result.setSamples(dataset.getNSamples()).setFeatures(dataset.getNFeatures()).setClasses(dataset.getNClasses());
            partial_result.setHyperparameters(experiment.getHyperParameters().get(dataset_name));
            for (int fold = 0; fold < data_size; ++fold) {
                partial_result.addScoreTest(data[fold]["score"]);
                partial_result.addScoreTrain(0.0);
                partial_result.addTimeTest(data[fold]["time"]);
                partial_result.addTimeTrain(data[fold]["time_train"]);
                score[fold] = data[fold]["score"].get<double>();
                time_test[fold] = data[fold]["time"].get<double>();
                time_train[fold] = data[fold]["time_train"].get<double>();
                nodes[fold] = data[fold]["nodes"].get<double>();
                leaves[fold] = data[fold]["leaves"].get<double>();
                depth[fold] = data[fold]["depth"].get<double>();
            }
            partial_result.setGraph(std::vector<std::string>());
            partial_result.setScoreTest(torch::mean(score).item<double>()).setScoreTrain(0.0);
            partial_result.setScoreTestStd(torch::std(score).item<double>()).setScoreTrainStd(0.0);
            partial_result.setTrainTime(torch::mean(time_train).item<double>()).setTestTime(torch::mean(time_test).item<double>());
            partial_result.setTrainTimeStd(torch::std(time_train).item<double>()).setTestTimeStd(torch::std(time_test).item<double>());
            partial_result.setNodes(torch::mean(nodes).item<double>()).setLeaves(torch::mean(leaves).item<double>()).setDepth(torch::mean(depth).item<double>());
            partial_result.setDataset(dataset_name).setNotes(std::vector<std::string>());
            partial_result.setConfusionMatrices(json::array());
            experiment.addResult(partial_result);
        }
        auto clf = Models::instance()->create(experiment.getModel());
        experiment.setModelVersion(clf->getVersion());
        computed_results = results;
    }
    json GridExperiment::store_result(std::vector<std::string>& names, Task_Result& result, json& results)
    {
        json json_result = {
            { "score", result.score },
            { "combination", result.idx_combination },
            { "fold", result.n_fold },
            { "time", result.time },
            { "time_train", result.time_train },
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
        Timer train_timer, test_timer;
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
        train_timer.start();
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
        auto train_time = train_timer.getDuration();
        //
        // Test model
        //
        test_timer.start();
        double score = clf->score(X_test, y_test);
        delete fold;
        auto test_time = test_timer.getDuration();
        //
        // Return the result
        //
        result->idx_dataset = task["idx_dataset"].get<int>();
        result->idx_combination = 0;
        result->score = score;
        result->n_fold = n_fold;
        result->time = test_time;
        result->time_train = train_time;
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