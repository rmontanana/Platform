#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <unistd.h>
#include <limits.h>
#include <torch/torch.h>
#include <folding.hpp>
#include "main/Models.h"
#include "common/Paths.h"
#include "common/Utils.h"
#include "GridExperiment.h"
#include "main/FoldRun.h"
#include <sstream>

#ifndef HOST_NAME_MAX
#include <sys/param.h>
#ifdef MAXHOSTNAMELEN
#define HOST_NAME_MAX MAXHOSTNAMELEN
#else
// 3. Fallback final si ninguna de las dos existe (255 es el límite POSIX estándar)
#define HOST_NAME_MAX 255 
#endif
#endif
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
        this->config.discretize_algo = experiment.getDiscretizationAlgorithm();
        this->config.n_folds = experiment.getNFolds();
        this->config.seeds = experiment.getRandomSeeds();
        this->config.quiet = experiment.isQuiet();
        this->config.no_train_score = experiment.isNoTrainScore();
        this->config.graph = experiment.doGraph();
        this->config.generate_fold_files = experiment.doGenerateFoldFiles();
    }
    FoldOptions GridExperiment::fold_options(Dataset& dataset)
    {
        FoldOptions options;
        options.model = config.model;
        if (config.score == "accuracy")
            options.score = score_t::ACCURACY;
        else if (config.score == "roc-auc-ovr")
            options.score = score_t::ROC_AUC_OVR;
        else
            throw std::runtime_error("Unknown score: " + config.score);
        options.no_train_score = config.no_train_score;
        options.discretized = config.discretize;
        options.graph = config.graph;
        options.smoothing = smooth_type;
        options.num_classes = dataset.getNClasses();
        options.labels = dataset.getLabels();
        return options;
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
            auto& dataset = datasets.getDataset(dataset_name);
            dataset.load();
            std::vector<FoldOutcome> folds;
            for (const auto& item : data) {
                FoldOutcome f;
                f.seed = item["seed"].get<int>();
                f.fold = item["fold"].get<int>();
                f.score_test = item["score"].get<double>();
                f.score_train = item["score_train"].get<double>();
                f.has_train = item["has_train"].get<bool>();
                f.time_test = item["time"].get<double>();
                f.time_train = item["time_train"].get<double>();
                f.nodes = item["nodes"].get<double>();
                f.edges = item["leaves"].get<double>();
                f.states = item["depth"].get<double>();
                f.cm_test = item["cm_test"].get<std::vector<int>>();
                f.cm_train = item["cm_train"].get<std::vector<int>>();
                f.notes = item["notes"].get<std::vector<std::string>>();
                f.graph = item["graph"].get<std::string>();
                folds.push_back(std::move(f));
            }
            // The tasks arrive in any order; b_main stores them seed x fold (Platform#8)
            sort_by_seed_fold(folds, config.seeds, config.n_folds);
            auto options = fold_options(dataset);
            auto partial_result = build_partial_result(folds, options, dataset_name, dataset.getNSamples(), dataset.getNFeatures(),
                dataset.getNClasses(), experiment.getHyperParameters().get(dataset_name));
            experiment.addResult(partial_result);
        }
        auto clf = Models::instance()->create(experiment.getModel());
        experiment.setModelVersion(clf->getVersion());
        computed_results = results;
    }
    json GridExperiment::store_result(std::vector<std::string>& names, Task_Result& result, Task_Payload& payload, json& results)
    {
        std::vector<std::string> notes;
        std::stringstream ss(payload.notes);
        for (std::string note; std::getline(ss, note, '\n');) {
            notes.push_back(note);
        }
        json json_result = {
            { "score", result.score },
            { "score_train", result.score_train },
            { "has_train", result.has_train != 0 },
            { "combination", result.idx_combination },
            { "seed", result.seed },
            { "fold", result.n_fold },
            { "time", result.time },
            { "time_train", result.time_train },
            { "dataset", result.idx_dataset },
            { "nodes", result.nodes },
            { "leaves", result.leaves },
            { "depth", result.depth },
            { "process", result.process },
            { "task", result.task },
            { "cm_test", payload.cm_test },
            { "cm_train", payload.cm_train },
            { "notes", notes },
            { "graph", payload.graph }
        };
        auto name = names[result.idx_dataset];
        if (!results.contains(name)) {
            results[name] = json::array();
        }
        results[name].push_back(json_result);
        return results;
    }
    void GridExperiment::consumer_go(struct ConfigGrid& config, struct ConfigMPI& config_mpi, json& tasks, int n_task, Datasets& datasets, Task_Result* result, Task_Payload& payload)
    {
        //
        // initialize
        //
        json task = tasks[n_task];
        auto dataset_name = task["dataset"].get<std::string>();
        auto seed = task["seed"].get<int>();
        auto n_fold = task["fold"].get<int>();
        bool stratified = config.stratified;
        //
        // Traza para localizar la tarea que provoca un fallo en un worker.
        // Se activa con BGRID_TRACE=1; sin esa variable no imprime nada, así
        // que la salida normal del experimento no cambia. La última línea que
        // deja un rank antes de morir dice en qué tarea y en qué fase estaba.
        //
        static const bool trace = std::getenv("BGRID_TRACE") != nullptr;
        auto trace_step = [&](const char* step) {
            if (!trace) return;
            char host[HOST_NAME_MAX + 1] = {};
            gethostname(host, sizeof(host) - 1);
            std::cerr << "[trace rank " << config_mpi.rank << "@" << host
                << " task=" << n_task << " " << dataset_name
                << " seed=" << seed << " fold=" << n_fold
                << "] " << step << std::endl;
            };
        trace_step("start");
        //
        // Generate the hyperparameters combinations
        //
        auto& dataset = datasets.getDataset(dataset_name);
        dataset.load();
        trace_step("dataset loaded");
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
        delete fold;
        auto [X_train, X_test, y_train, y_test] = dataset.getTrainTestTensors(train, test);
        auto states = dataset.getStates(); // Get the states of the features Once they are discretized
        if (config.generate_fold_files)
            generate_files(dataset_name, config.discretize, stratified, seed, n_fold, X_train, y_train, X_test, y_test, train, test);
        trace_step("tensors ready");
        //
        // Train and test the model exactly as b_main does
        //
        auto options = fold_options(dataset);
        auto& hyperparameters = experiment.getHyperParameters();
        auto progress = [&](char phase, bayesnet::status_t) {
            trace_step(phase == 'a' ? "fitting" : phase == 'b' ? "scoring train" : "scoring test");
            };
        auto outcome = run_fold(options, hyperparameters, dataset_name, X_train, y_train, X_test, y_test, features, className, states, progress);
        trace_step("scored");
        //
        // Return the result
        //
        result->idx_dataset = task["idx_dataset"].get<int>();
        result->idx_combination = 0;
        result->seed = seed;
        result->score = outcome.score_test;
        result->score_train = outcome.score_train;
        result->has_train = outcome.has_train ? 1 : 0;
        result->n_fold = n_fold;
        result->time = outcome.time_test;
        result->time_train = outcome.time_train;
        result->nodes = outcome.nodes;
        result->leaves = outcome.edges;
        result->depth = outcome.states;
        result->process = config_mpi.rank;
        result->task = n_task;
        result->n_classes = outcome.cm_test.empty() ? 0 : options.num_classes;
        payload.cm_test = std::move(outcome.cm_test);
        payload.cm_train = std::move(outcome.cm_train);
        payload.notes.clear();
        for (size_t i = 0; i < outcome.notes.size(); ++i) {
            if (i > 0)
                payload.notes += '\n';
            payload.notes += outcome.notes[i];
        }
        payload.graph = std::move(outcome.graph);
        result->notes_len = static_cast<int>(payload.notes.size());
        result->graph_len = static_cast<int>(payload.graph.size());
        //
        // Progress is now displayed by the producer when it receives this result
        //
    }
} /* namespace platform */