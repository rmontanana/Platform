#include <algorithm>
#include <set>
#include <stdexcept>
#include "FoldRun.h"

// The parts of FoldRun.h that do not train anything: ordering the folds and building the
// result of a dataset. Kept apart from FoldRun.cpp so the unit tests can link them without
// the model registry (pyclassifiers, Python).
namespace platform {
    namespace {
        double tensor_mean(const torch::Tensor& t) { return torch::mean(t).item<double>(); }
        double tensor_std(const torch::Tensor& t) { return torch::std(t).item<double>(); }
    }
    void sort_by_seed_fold(std::vector<FoldOutcome>& folds, const std::vector<int>& seeds, int n_folds)
    {
        auto seed_index = [&seeds](int seed) {
            auto it = std::find(seeds.begin(), seeds.end(), seed);
            if (it == seeds.end())
                throw std::runtime_error("Fold with seed " + std::to_string(seed) + ", which is not in the experiment");
            return static_cast<int>(std::distance(seeds.begin(), it));
            };
        std::set<std::pair<int, int>> seen;
        for (const auto& f : folds) {
            if (f.fold < 0 || f.fold >= n_folds)
                throw std::runtime_error("Fold number " + std::to_string(f.fold) + " out of range");
            if (!seen.insert({ seed_index(f.seed), f.fold }).second)
                throw std::runtime_error("Duplicated fold: seed " + std::to_string(f.seed) + " fold " + std::to_string(f.fold));
        }
        if (seen.size() != seeds.size() * static_cast<size_t>(n_folds))
            throw std::runtime_error("Expected " + std::to_string(seeds.size() * n_folds) + " folds, got " + std::to_string(seen.size()));
        std::stable_sort(folds.begin(), folds.end(), [&](const FoldOutcome& a, const FoldOutcome& b) {
            auto ia = seed_index(a.seed), ib = seed_index(b.seed);
            return ia != ib ? ia < ib : a.fold < b.fold;
            });
    }
    json confusion_matrix_json(const std::vector<int>& cm, int num_classes, const std::vector<std::string>& labels)
    {
        if (cm.size() != static_cast<size_t>(num_classes) * num_classes)
            throw std::runtime_error("Confusion matrix of " + std::to_string(cm.size()) + " cells for " + std::to_string(num_classes) + " classes");
        json output;
        for (int i = 0; i < num_classes; i++) {
            auto label = i < static_cast<int>(labels.size()) ? labels[i] : "Class " + std::to_string(i);
            output[label] = std::vector<int>(cm.begin() + i * num_classes, cm.begin() + (i + 1) * num_classes);
        }
        return output;
    }
    PartialResult build_partial_result(const std::vector<FoldOutcome>& folds, const FoldOptions& options,
        const std::string& dataset, long samples, long n_features, int classes, const json& hyperparameters)
    {
        auto partial_result = PartialResult();
        partial_result.setSamples(samples).setFeatures(n_features).setClasses(classes);
        partial_result.setHyperparameters(hyperparameters);
        int n = static_cast<int>(folds.size());
        auto score_test = torch::zeros({ n }, torch::kFloat64);
        auto score_train = torch::zeros({ n }, torch::kFloat64);
        auto train_time = torch::zeros({ n }, torch::kFloat64);
        auto test_time = torch::zeros({ n }, torch::kFloat64);
        auto nodes = torch::zeros({ n }, torch::kFloat64);
        auto edges = torch::zeros({ n }, torch::kFloat64);
        auto num_states = torch::zeros({ n }, torch::kFloat64);
        json confusion_matrices = json::array();
        json confusion_matrices_train = json::array();
        std::vector<std::string> notes;
        std::vector<std::string> graphs;
        for (int item = 0; item < n; ++item) {
            const auto& f = folds[item];
            score_test[item] = f.score_test;
            score_train[item] = f.score_train;
            train_time[item] = f.time_train;
            test_time[item] = f.time_test;
            nodes[item] = f.nodes;
            edges[item] = f.edges;
            num_states[item] = f.states;
            for (const auto& note : f.notes) {
                notes.push_back("Seed: " + std::to_string(f.seed) + " Fold: " + std::to_string(f.fold) + ": " + note);
            }
            if (options.discretized) {
                confusion_matrices.push_back(confusion_matrix_json(f.cm_test, options.num_classes, options.labels));
                if (f.has_train)
                    confusion_matrices_train.push_back(confusion_matrix_json(f.cm_train, options.num_classes, options.labels));
            }
            partial_result.addScoreTrain(f.score_train);
            partial_result.addScoreTest(f.score_test);
            partial_result.addTimeTrain(f.time_train);
            partial_result.addTimeTest(f.time_test);
            if (options.graph)
                graphs.push_back(f.graph);
        }
        partial_result.setGraph(graphs);
        partial_result.setScoreTest(tensor_mean(score_test)).setScoreTrain(tensor_mean(score_train));
        partial_result.setScoreTestStd(tensor_std(score_test)).setScoreTrainStd(tensor_std(score_train));
        partial_result.setTrainTime(tensor_mean(train_time)).setTestTime(tensor_mean(test_time));
        partial_result.setTestTimeStd(tensor_std(test_time)).setTrainTimeStd(tensor_std(train_time));
        partial_result.setNodes(tensor_mean(nodes)).setLeaves(tensor_mean(edges)).setDepth(tensor_mean(num_states));
        partial_result.setDataset(dataset).setNotes(notes);
        partial_result.setConfusionMatrices(confusion_matrices);
        if (!options.no_train_score)
            partial_result.setConfusionMatricesTrain(confusion_matrices_train);
        return partial_result;
    }
}
