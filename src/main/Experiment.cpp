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
        result.setSchemaVersion("1.0");
        result.check();
        result.save();
        std::cout << "Result saved in " << Paths::results() << result.getFilename() << std::endl;
    }
    void Experiment::report()
    {
        ReportConsole report(result.getJson());
        report.show();
        if (filesToTest.size() == 1) {
            std::cout << report.showClassificationReport(Colors::BLUE());
        }
    }
    void Experiment::show()
    {
        std::cout << result.getJson().dump(4) << std::endl;
    }
    void Experiment::saveGraph()
    {
        std::cout << "Saving graphs..." << std::endl;
        auto data = result.getJson();
        for (const auto& item : data["results"]) {
            auto graphs = item["graph"];
            int i = 0;
            for (const auto& graph : graphs) {
                i++;
                auto fileName = Paths::graphs() + result.getFilename() + "_graph_" + item["dataset"].get<std::string>() + "_" + std::to_string(i) + ".dot";
                auto file = std::ofstream(fileName);
                file << graph.get<std::string>();
                file.close();
                std::cout << "Graph saved in " << fileName << std::endl;
            }
        }
    }
    Experiment& Experiment::setSmoothSrategy(const std::string& smooth_strategy)
    {
        this->smooth_strategy = smooth_strategy;
        this->result.setSmoothStrategy(smooth_strategy);
        if (smooth_strategy == "ORIGINAL")
            smooth_type = bayesnet::Smoothing_t::ORIGINAL;
        else if (smooth_strategy == "LAPLACE")
            smooth_type = bayesnet::Smoothing_t::LAPLACE;
        else if (smooth_strategy == "CESTNIK")
            smooth_type = bayesnet::Smoothing_t::CESTNIK;
        else {
            std::cerr << "Experiment: Unknown smoothing strategy: " << smooth_strategy << std::endl;
            exit(1);
        }
        return *this;
    }
    void Experiment::go()
    {
        for (auto fileName : filesToTest) {
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
            int nc = 4 + 3 * nfolds + (nfolds >= 10 ? nfolds - 10 + 1 : 0);
            std::cout << Colors::GREEN() << left << "  #  " << setw(max_name) << "Dataset" << " #Samp #Feat Seed Status" << string(nc - 6, ' ') << setw(11) << " Time" << " Score" << std::endl;
            std::cout << " --- " << string(max_name, '-') << " ----- ----- ---- " << string(nc, '-') << " ----------" << " ---------";
            std::cout << Colors::RESET() << std::endl;
        }
        int num = 0;
        for (auto fileName : filesToTest) {
            if (!quiet)
                std::cout << " " << setw(3) << right << num++ << " " << setw(max_name) << left << fileName << right << flush;
            cross_validation(fileName);
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
    score_t Experiment::parse_score() const
    {
        if (result.getScoreName() == "accuracy")
            return score_t::ACCURACY;
        if (result.getScoreName() == "roc-auc-ovr")
            return score_t::ROC_AUC_OVR;
        throw std::runtime_error("Unknown score: " + result.getScoreName());
    }
    void showProgress(int fold, const std::string& color, const std::string& phase)
    {
        int nc = fold >= 10 ? 5 : 4;
        std::string prefix = phase == "-" ? "" : std::string(nc, '\b');
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
    void Experiment::cross_validation(const std::string& fileName)
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
        auto score_test = torch::zeros({ nResults }, torch::kFloat64);
        auto score_train = torch::zeros({ nResults }, torch::kFloat64);
        auto train_time = torch::zeros({ nResults }, torch::kFloat64);
        auto test_time = torch::zeros({ nResults }, torch::kFloat64);
        auto nodes = torch::zeros({ nResults }, torch::kFloat64);
        auto edges = torch::zeros({ nResults }, torch::kFloat64);
        auto num_states = torch::zeros({ nResults }, torch::kFloat64);
        json confusion_matrices = json::array();
        json confusion_matrices_train = json::array();
        std::vector<std::string> notes;
        std::vector<std::string> graphs;
        Timer train_timer, test_timer, seed_timer;
        int item = 0;
        bool first_seed = true;
        //
        // Loop over random seeds
        //
        auto score = parse_score();
        for (auto seed : randomSeeds) {
            seed_timer.start();
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
                if (!quiet)
                    showProgress(nfold + 1, getColor(clf->getStatus()), "-");
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
                clf->fit(X_train, y_train, features, className, states, smooth_type);
                if (!quiet)
                    showProgress(nfold + 1, getColor(clf->getStatus()), "b");
                auto clf_notes = clf->getNotes();
                std::transform(clf_notes.begin(), clf_notes.end(), std::back_inserter(notes), [nfold](const std::string& note)
                    { return "Fold " + std::to_string(nfold) + ": " + note; });
                nodes[item] = clf->getNumberOfNodes();
                edges[item] = clf->getNumberOfEdges();
                num_states[item] = clf->getNumberOfStates();
                train_time[item] = train_timer.getDuration();
                double score_train_value = 0.0;
                //
                // Score train
                //
                if (!no_train_score) {
                    auto y_proba_train = clf->predict_proba(X_train);
                    Scores scores(y_train, y_proba_train, num_classes, labels);
                    score_train_value = score == score_t::ACCURACY ? scores.accuracy() : scores.auc();
                    if (discretized)
                        confusion_matrices_train.push_back(scores.get_confusion_matrix_json(true));
                }
                //
                // Test model
                //
                if (!quiet)
                    showProgress(nfold + 1, getColor(clf->getStatus()), "c");
                test_timer.start();
                // auto y_predict = clf->predict(X_test);
                auto y_proba_test = clf->predict_proba(X_test);
                Scores scores(y_test, y_proba_test, num_classes, labels);
                auto score_test_value = score == score_t::ACCURACY ? scores.accuracy() : scores.auc();
                test_time[item] = test_timer.getDuration();
                score_train[item] = score_train_value;
                score_test[item] = score_test_value;
                if (discretized)
                    confusion_matrices.push_back(scores.get_confusion_matrix_json(true));
                if (!quiet)
                    std::cout << "\b\b\b, " << flush;
                //
                // Store results and times in std::vector
                //
                partial_result.addScoreTrain(score_train_value);
                partial_result.addScoreTest(score_test_value);
                partial_result.addTimeTrain(train_time[item].item<double>());
                partial_result.addTimeTest(test_time[item].item<double>());
                item++;
                if (graph) {
                    std::string result = "";
                    for (const auto& line : clf->graph()) {
                        result += line + "\n";
                    }
                    graphs.push_back(result);
                }
            }
            if (!quiet) {
                seed_timer.stop();
                std::cout << "end. " << std::setw(10) << std::right << seed_timer.getDurationString();
            }
            delete fold;
        }
        // Show Results
        if (!quiet)
            std::cout << " " << setw(9) << right << std::fixed << std::setprecision(7) << torch::mean(score_test).item<double>();
        //
        // Store result totals in Result
        //
        partial_result.setGraph(graphs);
        partial_result.setScoreTest(torch::mean(score_test).item<double>()).setScoreTrain(torch::mean(score_train).item<double>());
        partial_result.setScoreTestStd(torch::std(score_test).item<double>()).setScoreTrainStd(torch::std(score_train).item<double>());
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