#ifndef FOLDRUN_H
#define FOLDRUN_H
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include "bayesnet/BaseClassifier.h"
#include "HyperParameters.h"
#include "PartialResult.h"
#include "Experiment.h"

//
// One fold of a cross validation and the result of a dataset built from its folds.
// b_main (Experiment::cross_validation) and b_grid experiment (GridExperiment) both go
// through these functions, so the two programs train, predict, time and store a fold
// in exactly the same way (Platform#8, Platform#9).
//
namespace platform {
    using json = nlohmann::ordered_json;
    // What a fold needs besides its data.
    struct FoldOptions {
        std::string model;
        score_t score = score_t::ACCURACY;
        bool no_train_score = false;
        bool discretized = false;   // confusion matrices are stored only for discretized datasets (as b_main always did)
        bool graph = false;
        bayesnet::Smoothing_t smoothing = bayesnet::Smoothing_t::ORIGINAL;
        int num_classes = 0;
        std::vector<std::string> labels;
    };
    // Everything a fold produces. It travels from the b_grid workers to the manager.
    struct FoldOutcome {
        int seed = 0;
        int fold = 0;
        double score_test = 0.0;
        double score_train = 0.0;   // 0 when the train score is not computed (--no-train-score)
        bool has_train = false;
        double time_train = 0.0;    // fit only
        double time_test = 0.0;     // predict_proba of the test partition only
        double nodes = 0.0, edges = 0.0, states = 0.0;
        std::vector<int> cm_test, cm_train; // num_classes x num_classes, row major (actual, predicted); empty if not discretized
        std::vector<std::string> notes;     // classifier notes, without the "Seed: s Fold: f: " prefix
        std::string graph;                  // graphviz text, only when FoldOptions::graph
        std::string version;                // model version
    };
    // Called before each phase: 'a' fit, 'b' train score, 'c' test score.
    using FoldProgress = std::function<void(char phase, bayesnet::status_t status)>;

    FoldOutcome run_fold(const FoldOptions& options, HyperParameters& hyperparameters, const std::string& dataset,
        torch::Tensor& X_train, torch::Tensor& y_train, torch::Tensor& X_test, torch::Tensor& y_test,
        const std::vector<std::string>& features, const std::string& className,
        std::map<std::string, std::vector<int>>& states, const FoldProgress& progress = nullptr);
    // Puts the folds in the order b_main produces them: seeds in the order given, folds ascending.
    // Throws std::runtime_error unless there is exactly one fold for every seed x fold.
    void sort_by_seed_fold(std::vector<FoldOutcome>& folds, const std::vector<int>& seeds, int n_folds);
    // The result of a dataset from its folds, already in seed x fold order.
    PartialResult build_partial_result(const std::vector<FoldOutcome>& folds, const FoldOptions& options,
        const std::string& dataset, long samples, long n_features, int classes, const json& hyperparameters);
    // Confusion matrix with the class labels as keys, as Scores::get_confusion_matrix_json(true).
    json confusion_matrix_json(const std::vector<int>& cm, int num_classes, const std::vector<std::string>& labels);
    // Dumps the partitions of a fold to datasets_experiment/ (--generate-fold-files).
    void generate_files(const std::string& fileName, bool discretize, bool stratified, int seed, int nfold,
        torch::Tensor X_train, torch::Tensor y_train, torch::Tensor X_test, torch::Tensor y_test,
        std::vector<int>& train, std::vector<int>& test);
}
#endif
