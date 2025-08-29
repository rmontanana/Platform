#include <iostream>
#include <cstddef>
#include <torch/torch.h>
#include <folding.hpp>
#include "main/Models.h"
#include "common/Paths.h"
#include "common/Utils.h"
#include "GridExperiment.h"

namespace platform {
    // GridExperiment::GridExperiment(argparse::ArgumentParser& program, struct ConfigGrid& config) : arguments(program), GridBase(config)
    GridExperiment::GridExperiment(ArgumentsExperiment& program, struct ConfigGrid& config) : arguments(program), GridBase(config)
    {
        experiment = arguments.initializedExperiment();
        filesToTest = arguments.getFilesToTest();
        saveResults = arguments.haveToSaveResults();
        this->config.model = experiment.getModel();
        this->config.score = experiment.getScore();
        this->config.discretize = experiment.isDiscretized();
        this->config.stratified = experiment.isStratified();
        this->config.smooth_strategy = experiment.getSmoothStrategy();
        this->config.n_folds = experiment.getNFolds();
        this->config.seeds = experiment.getRandomSeeds();
        this->config.quiet = experiment.isQuiet();
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
        // Progress is now displayed by the producer when it receives this result
        //
    }
} /* namespace platform */