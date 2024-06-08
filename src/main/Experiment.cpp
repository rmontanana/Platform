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
        std::cout << "Result saved in " << Paths::results() << result.getFilename() << std::endl;
    }
    void Experiment::report(bool classification_report)
    {
        ReportConsole report(result.getJson());
        report.show();
        if (classification_report) {
            std::cout << report.showClassificationReport(Colors::BLUE());
        }
    }
    void Experiment::show()
    {
        std::cout << result.getJson().dump(4) << std::endl;
    }
    void Experiment::go(std::vector<std::string> filesToProcess, bool quiet, bool no_train_score, bool generate_fold_files)
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
            cross_validation(fileName, quiet, no_train_score, generate_fold_files);
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
    void generate_files(const std::string& fileName, bool discretize, bool stratified, int seed, int nfold, torch::Tensor X_train, torch::Tensor y_train, torch::Tensor X_test, torch::Tensor y_test, std::vector<int>& train, std::vector<int>& test)
    {
        std::string file_name = Paths::experiment_file(fileName, discretize, stratified, seed, nfold);
        auto file = std::ofstream(file_name);
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
    void Experiment::cross_validation(const std::string& fileName, bool quiet, bool no_train_score, bool generate_fold_files)
    {
        //
        // Load dataset and prepare data
        //
        auto datasets = Datasets(discretized, Paths::datasets(), discretization_algo);
        auto& dataset = datasets.getDataset(fileName);
        dataset.load();
        auto [X, y] = dataset.getTensors(); // Only need y for folding
        auto features = dataset.getFeatures();
        auto n_features = dataset.getNFeatures();
        auto n_samples = dataset.getNSamples();
        auto className = dataset.getClassName();
        auto labels = dataset.getLabels();
        int num_classes = dataset.getNClasses();
        if (!quiet) {
            std::cout << " " << setw(5) << n_samples << " " << setw(5) << n_features << flush;
        }
        //
        // Prepare Result
        //
        auto partial_result = PartialResult();
        partial_result.setSamples(n_samples).setFeatures(n_features).setClasses(num_classes);
        partial_result.setHyperparameters(hyperparameters.get(fileName));
        //
        // Initialize results std::vectors
        //
        int nResults = nfolds * static_cast<int>(randomSeeds.size());
        auto accuracy_test = torch::zeros({ nResults }, torch::kFloat64);
        auto accuracy_train = torch::zeros({ nResults }, torch::kFloat64);
        auto train_time = torch::zeros({ nResults }, torch::kFloat64);
        auto test_time = torch::zeros({ nResults }, torch::kFloat64);
        auto nodes = torch::zeros({ nResults }, torch::kFloat64);
        auto edges = torch::zeros({ nResults }, torch::kFloat64);
        auto num_states = torch::zeros({ nResults }, torch::kFloat64);
        json confusion_matrices = json::array();
        json confusion_matrices_train = json::array();
        std::vector<std::string> notes;
        Timer train_timer, test_timer;
        int item = 0;
        bool first_seed = true;
        //
        // Loop over random seeds
        //
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
                fold = new folding::KFold(nfolds, n_samples, seed);
            //
            // Loop over folds
            //
            for (int nfold = 0; nfold < nfolds; nfold++) {
                auto clf = Models::instance()->create(result.getModel());
                setModelVersion(clf->getVersion());
                auto valid = clf->getValidHyperparameters();
                hyperparameters.check(valid, fileName);
                clf->setHyperparameters(hyperparameters.get(fileName));
                //
                // Split train - test dataset
                //
                train_timer.start();
                auto [train, test] = fold->getFold(nfold);
                auto [X_train, X_test, y_train, y_test] = dataset.getTrainTestTensors(train, test);
                auto states = dataset.getStates(); // Get the states of the features Once they are discretized
                if (generate_fold_files)
                    generate_files(fileName, discretized, stratified, seed, nfold, X_train, y_train, X_test, y_test, train, test);
                if (!quiet)
                    showProgress(nfold + 1, getColor(clf->getStatus()), "a");
                //
                // Train model
                //
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
                //
                // Score train
                //
                if (!no_train_score) {
                    auto y_predict = clf->predict(X_train);
                    Scores scores(y_train, y_predict, num_classes, labels);
                    accuracy_train_value = scores.accuracy();
                    confusion_matrices_train.push_back(scores.get_confusion_matrix_json(true));
                }
                //
                // Test model
                //
                if (!quiet)
                    showProgress(nfold + 1, getColor(clf->getStatus()), "c");
                test_timer.start();
                auto y_predict = clf->predict(X_test);
                Scores scores(y_test, y_predict, num_classes, labels);
                auto accuracy_test_value = scores.accuracy();
                test_time[item] = test_timer.getDuration();
                accuracy_train[item] = accuracy_train_value;
                accuracy_test[item] = accuracy_test_value;
                confusion_matrices.push_back(scores.get_confusion_matrix_json(true));
                if (!quiet)
                    std::cout << "\b\b\b, " << flush;
                //
                // Store results and times in std::vector
                //
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
        //
        // Store result totals in Result
        //
        partial_result.setScoreTest(torch::mean(accuracy_test).item<double>()).setScoreTrain(torch::mean(accuracy_train).item<double>());
        partial_result.setScoreTestStd(torch::std(accuracy_test).item<double>()).setScoreTrainStd(torch::std(accuracy_train).item<double>());
        partial_result.setTrainTime(torch::mean(train_time).item<double>()).setTestTime(torch::mean(test_time).item<double>());
        partial_result.setTestTimeStd(torch::std(test_time).item<double>()).setTrainTimeStd(torch::std(train_time).item<double>());
        partial_result.setNodes(torch::mean(nodes).item<double>()).setLeaves(torch::mean(edges).item<double>()).setDepth(torch::mean(num_states).item<double>());
        partial_result.setDataset(fileName).setNotes(notes);
        partial_result.setConfusionMatrices(confusion_matrices);
        if (!no_train_score)
            partial_result.setConfusionMatricesTrain(confusion_matrices_train);
        addResult(partial_result);
    }
}