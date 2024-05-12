#include "common/Datasets.h"
#include "reports/ReportConsole.h"
#include "common/Paths.h"
#include "Models.h"
#include "Scores.h"
#include "Experiment.h"
namespace platform {
    using json = nlohmann::ordered_json;

    void Experiment::saveResult()
    {
        result.save();
    }
    void Experiment::report(bool classification_report)
    {
        ReportConsole report(result.getJson());
        report.show();
        if (classification_report) {
            std::cout << Colors::BLUE() << report.showClassificationReport() << Colors::RESET();
        }
    }
    void Experiment::show()
    {
        std::cout << result.getJson().dump(4) << std::endl;
    }
    void Experiment::go(std::vector<std::string> filesToProcess, bool quiet, bool no_train_score)
    {
        for (auto fileName : filesToProcess) {
            if (fileName.size() > max_name)
                max_name = fileName.size();
        }
        std::cout << Colors::MAGENTA() << "*** Starting experiment: " << result.getTitle() << " ***" << Colors::RESET() << std::endl << std::endl;
        auto clf = Models::instance()->create(result.getModel());
        auto version = clf->getVersion();
        std::cout << Colors::BLUE() << " Using " << result.getModel() << " ver. " << version << std::endl << std::endl;
        if (!quiet) {
            std::cout << Colors::GREEN() << " Status Meaning" << std::endl;
            std::cout << " ------ --------------------------------" << Colors::RESET() << std::endl;
            std::cout << " ( " << Colors::GREEN() << "a" << Colors::RESET() << " )  Fitting model with train dataset" << std::endl;
            std::cout << " ( " << Colors::GREEN() << "b" << Colors::RESET() << " )  Scoring train dataset" << std::endl;
            std::cout << " ( " << Colors::GREEN() << "c" << Colors::RESET() << " )  Scoring test dataset" << std::endl << std::endl;
            std::cout << Colors::YELLOW() << "Note: fold number in this color means fitting had issues such as not using all features in BoostAODE classifier" << std::endl << std::endl;
            std::cout << Colors::GREEN() << left << "  #  " << setw(max_name) << "Dataset" << " #Samp #Feat Seed Status" << std::endl;
            std::cout << " --- " << string(max_name, '-') << " ----- ----- ---- " << string(4 + 3 * nfolds, '-') << Colors::RESET() << std::endl;
        }
        int num = 0;
        for (auto fileName : filesToProcess) {
            if (!quiet)
                std::cout << " " << setw(3) << right << num++ << " " << setw(max_name) << left << fileName << right << flush;
            cross_validation(fileName, quiet, no_train_score);
            if (!quiet)
                std::cout << std::endl;
        }
        if (!quiet)
            std::cout << std::endl;
    }
    std::string getColor(bayesnet::status_t status)
    {
        switch (status) {
            case bayesnet::NORMAL:
                return Colors::GREEN();
            case bayesnet::WARNING:
                return Colors::YELLOW();
            case bayesnet::ERROR:
                return Colors::RED();
            default:
                return Colors::RESET();
        }
    }

    void showProgress(int fold, const std::string& color, const std::string& phase)
    {
        std::string prefix = phase == "a" ? "" : "\b\b\b\b";
        std::cout << prefix << color << fold << Colors::RESET() << "(" << color << phase << Colors::RESET() << ")" << flush;

    }
    void Experiment::cross_validation(const std::string& fileName, bool quiet, bool no_train_score)
    {
        auto datasets = Datasets(discretized, Paths::datasets());
        // Get dataset
        auto [X, y] = datasets.getTensors(fileName);
        auto states = datasets.getStates(fileName);
        auto features = datasets.getFeatures(fileName);
        auto samples = datasets.getNSamples(fileName);
        auto className = datasets.getClassName(fileName);
        auto labels = datasets.getLabels(fileName);
        if (!quiet) {
            std::cout << " " << setw(5) << samples << " " << setw(5) << features.size() << flush;
        }
        // Prepare Result
        auto partial_result = PartialResult();
        auto [values, counts] = at::_unique(y);
        partial_result.setSamples(X.size(1)).setFeatures(X.size(0)).setClasses(values.size(0));
        partial_result.setHyperparameters(hyperparameters.get(fileName));
        // Initialize results std::vectors
        int nResults = nfolds * static_cast<int>(randomSeeds.size());
        auto accuracy_test = torch::zeros({ nResults }, torch::kFloat64);
        auto accuracy_train = torch::zeros({ nResults }, torch::kFloat64);
        auto train_time = torch::zeros({ nResults }, torch::kFloat64);
        auto test_time = torch::zeros({ nResults }, torch::kFloat64);
        auto nodes = torch::zeros({ nResults }, torch::kFloat64);
        auto edges = torch::zeros({ nResults }, torch::kFloat64);
        auto num_states = torch::zeros({ nResults }, torch::kFloat64);
        json confusion_matrices = json::array();
        std::vector<std::string> notes;
        Timer train_timer, test_timer;
        int item = 0;
        bool first_seed = true;
        for (auto seed : randomSeeds) {
            if (!quiet) {
                string prefix = " ";
                if (!first_seed) {
                    prefix = "\n" + string(18 + max_name, ' ');
                }
                std::cout << prefix << setw(4) << right << seed << " " << flush;
                first_seed = false;
            }
            folding::Fold* fold;
            if (stratified)
                fold = new folding::StratifiedKFold(nfolds, y, seed);
            else
                fold = new folding::KFold(nfolds, y.size(0), seed);
            for (int nfold = 0; nfold < nfolds; nfold++) {
                auto clf = Models::instance()->create(result.getModel());
                setModelVersion(clf->getVersion());
                auto valid = clf->getValidHyperparameters();
                hyperparameters.check(valid, fileName);
                clf->setHyperparameters(hyperparameters.get(fileName));
                // Split train - test dataset
                train_timer.start();
                auto [train, test] = fold->getFold(nfold);
                auto train_t = torch::tensor(train);
                auto test_t = torch::tensor(test);
                auto X_train = X.index({ "...", train_t });
                auto y_train = y.index({ train_t });
                auto X_test = X.index({ "...", test_t });
                auto y_test = y.index({ test_t });
                if (!quiet)
                    showProgress(nfold + 1, getColor(clf->getStatus()), "a");
                // Train model
                clf->fit(X_train, y_train, features, className, states);
                if (!quiet)
                    showProgress(nfold + 1, getColor(clf->getStatus()), "b");
                auto clf_notes = clf->getNotes();
                std::transform(clf_notes.begin(), clf_notes.end(), std::back_inserter(notes), [nfold](const std::string& note)
                    { return "Fold " + std::to_string(nfold) + ": " + note; });
                nodes[item] = clf->getNumberOfNodes();
                edges[item] = clf->getNumberOfEdges();
                num_states[item] = clf->getNumberOfStates();
                train_time[item] = train_timer.getDuration();
                double accuracy_train_value = 0.0;
                // Score train
                if (!no_train_score)
                    accuracy_train_value = clf->score(X_train, y_train);
                // Test model
                if (!quiet)
                    showProgress(nfold + 1, getColor(clf->getStatus()), "c");
                test_timer.start();
                auto y_predict = clf->predict(X_test);
                Scores scores(y_test, y_predict, states[className].size(), labels);
                auto accuracy_test_value = scores.accuracy();
                test_time[item] = test_timer.getDuration();
                accuracy_train[item] = accuracy_train_value;
                accuracy_test[item] = accuracy_test_value;
                confusion_matrices.push_back(scores.get_confusion_matrix_json(true));
                if (!quiet)
                    std::cout << "\b\b\b, " << flush;
                // Store results and times in std::vector
                partial_result.addScoreTrain(accuracy_train_value);
                partial_result.addScoreTest(accuracy_test_value);
                partial_result.addTimeTrain(train_time[item].item<double>());
                partial_result.addTimeTest(test_time[item].item<double>());
                item++;
            }
            if (!quiet)
                std::cout << "end. " << flush;
            delete fold;
        }
        partial_result.setScoreTest(torch::mean(accuracy_test).item<double>()).setScoreTrain(torch::mean(accuracy_train).item<double>());
        partial_result.setScoreTestStd(torch::std(accuracy_test).item<double>()).setScoreTrainStd(torch::std(accuracy_train).item<double>());
        partial_result.setTrainTime(torch::mean(train_time).item<double>()).setTestTime(torch::mean(test_time).item<double>());
        partial_result.setTestTimeStd(torch::std(test_time).item<double>()).setTrainTimeStd(torch::std(train_time).item<double>());
        partial_result.setNodes(torch::mean(nodes).item<double>()).setLeaves(torch::mean(edges).item<double>()).setDepth(torch::mean(num_states).item<double>());
        partial_result.setDataset(fileName).setNotes(notes);
        partial_result.setConfusionMatrices(confusion_matrices);
        addResult(partial_result);
    }
}