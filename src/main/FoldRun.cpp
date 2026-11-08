#include <algorithm>
#include <fstream>
#include <stdexcept>
#include "common/Paths.h"
#include "common/Timer.hpp"
#include "Models.h"
#include "Scores.h"
#include "FoldRun.h"

namespace platform {
    namespace {
        double pick_score(Scores& scores, score_t score)
        {
            return score == score_t::ACCURACY ? scores.accuracy() : scores.auc();
        }
        std::vector<int> matrix_cells(const torch::Tensor& cm)
        {
            auto c = cm.contiguous().to(torch::kInt32);
            return std::vector<int>(c.data_ptr<int>(), c.data_ptr<int>() + c.numel());
        }
    }
    FoldOutcome run_fold(const FoldOptions& options, HyperParameters& hyperparameters, const std::string& dataset,
        torch::Tensor& X_train, torch::Tensor& y_train, torch::Tensor& X_test, torch::Tensor& y_test,
        const std::vector<std::string>& features, const std::string& className,
        std::map<std::string, std::vector<int>>& states, const FoldProgress& progress)
    {
        FoldOutcome out;
        auto clf = Models::instance()->create(options.model);
        out.version = clf->getVersion();
        auto valid = clf->getValidHyperparameters();
        hyperparameters.check(valid, dataset);
        clf->setHyperparameters(hyperparameters.get(dataset));
        //
        // Train: the time is the fit alone
        //
        if (progress) progress('a', clf->getStatus());
        Timer timer;
        timer.start();
        clf->fit(X_train, y_train, features, className, states, options.smoothing);
        out.time_train = timer.getDuration();
        out.notes = clf->getNotes();
        out.nodes = clf->getNumberOfNodes();
        out.edges = clf->getNumberOfEdges();
        out.states = clf->getNumberOfStates();
        //
        // Score train
        //
        if (!options.no_train_score) {
            if (progress) progress('b', clf->getStatus());
            auto y_proba_train = clf->predict_proba(X_train);
            Scores scores(y_train, y_proba_train, options.num_classes, options.labels);
            out.score_train = pick_score(scores, options.score);
            out.has_train = true;
            if (options.discretized)
                out.cm_train = matrix_cells(scores.get_confusion_matrix());
        }
        //
        // Test: the time is predict_proba alone; scores and confusion matrix come after
        //
        if (progress) progress('c', clf->getStatus());
        timer.start();
        auto y_proba_test = clf->predict_proba(X_test);
        out.time_test = timer.getDuration();
        Scores scores(y_test, y_proba_test, options.num_classes, options.labels);
        out.score_test = pick_score(scores, options.score);
        if (options.discretized)
            out.cm_test = matrix_cells(scores.get_confusion_matrix());
        if (options.graph) {
            for (const auto& line : clf->graph()) {
                out.graph += line + "\n";
            }
        }
        return out;
    }
    void generate_files(const std::string& fileName, bool discretize, bool stratified, int seed, int nfold, torch::Tensor X_train, torch::Tensor y_train, torch::Tensor X_test, torch::Tensor y_test, std::vector<int>& train, std::vector<int>& test)
    {
        std::string file_name = Paths::experiment_file(fileName, discretize, stratified, seed, nfold);
        auto file = std::ofstream(file_name);
        if (!file.is_open())
            throw std::runtime_error("Cannot write the fold file " + file_name + " (does the folder exist?)");
        json output;
        output["seed"] = seed;
        output["nfold"] = nfold;
        output["X_train"] = json::array();
        auto n = X_train.size(1);
        for (int i = 0; i < X_train.size(0); i++) {
            if (X_train.dtype() == torch::kFloat32) {
                auto xvf_ptr = X_train.index({ i }).data_ptr<float>();
                auto feature = std::vector<float>(xvf_ptr, xvf_ptr + n);
                output["X_train"].push_back(feature);
            } else {
                auto feature = std::vector<int>(X_train.index({ i }).data_ptr<int>(), X_train.index({ i }).data_ptr<int>() + n);
                output["X_train"].push_back(feature);
            }
        }
        output["y_train"] = std::vector<int>(y_train.data_ptr<int>(), y_train.data_ptr<int>() + n);
        output["X_test"] = json::array();
        n = X_test.size(1);
        for (int i = 0; i < X_test.size(0); i++) {
            if (X_train.dtype() == torch::kFloat32) {
                auto xvf_ptr = X_test.index({ i }).data_ptr<float>();
                auto feature = std::vector<float>(xvf_ptr, xvf_ptr + n);
                output["X_test"].push_back(feature);
            } else {
                auto feature = std::vector<int>(X_test.index({ i }).data_ptr<int>(), X_test.index({ i }).data_ptr<int>() + n);
                output["X_test"].push_back(feature);
            }
        }
        output["y_test"] = std::vector<int>(y_test.data_ptr<int>(), y_test.data_ptr<int>() + n);
        output["train"] = train;
        output["test"] = test;
        file << output.dump(4);
        file.close();
    }
}
