#include "common/Datasets.h"
#include "reports/ReportConsole.h"
#include "common/Paths.h"
#include "Models.h"
#include "Scores.h"
#include "Experiment.h"
#include "FoldRun.h"
namespace platform {
    using json = nlohmann::ordered_json;

    void Experiment::saveResult(const std::string& path)
    {
        result.setSchemaVersion("1.0");
        result.check();
        result.save(path);
        std::cout << "Result saved in " << path << result.getFilename() << std::endl;
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
            std::cout << Colors::GREEN() << std::left << "  #  " << std::setw(max_name) << "Dataset" << " #Samp. #Feat  k  Seed Status" << std::string(nc - 6, ' ') << std::setw(11) << " Time" << " Score" << std::endl;
            std::cout << " --- " << std::string(max_name, '-') << " ------ ----- --- ---- " << std::string(nc, '-') << " ----------" << " ---------";
            std::cout << Colors::RESET() << std::endl;
        }
        int num = 0;
        for (auto fileName : filesToTest) {
            if (!quiet)
                std::cout << " " << std::setw(3) << std::right << num++ << " " << std::setw(max_name) << std::left << fileName << std::right << std::flush;
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
        std::cout << prefix << color << fold << Colors::RESET() << "(" << color << phase << Colors::RESET() << ")" << std::flush;

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
            std::cout << " " << std::setw(6) << n_samples << " " << std::setw(5) << n_features << " " << std::setw(3) << num_classes << std::flush;
        }
        //
        // Every fold goes through run_fold and the result through build_partial_result,
        // the same functions b_grid experiment uses (Platform#8, Platform#9).
        //
        FoldOptions options;
        options.model = result.getModel();
        options.score = parse_score();
        options.no_train_score = no_train_score;
        options.discretized = discretized;
        options.graph = graph;
        options.smoothing = smooth_type;
        options.num_classes = num_classes;
        options.labels = labels;
        std::vector<FoldOutcome> outcomes;
        Timer seed_timer;
        bool first_seed = true;
        //
        // Loop over random seeds
        //
        for (auto seed : randomSeeds) {
            seed_timer.start();
            double seed_score_sum = 0.0;
            if (!quiet) {
                std::string prefix = " ";
                if (!first_seed) {
                    prefix = "\n" + std::string(23 + max_name, ' ');
                }
                std::cout << prefix << std::setw(4) << std::right << seed << " " << std::flush;
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
                if (!quiet)
                    showProgress(nfold + 1, getColor(bayesnet::NORMAL), "-");
                //
                // Split train - test dataset
                //
                auto [train, test] = fold->getFold(nfold);
                auto [X_train, X_test, y_train, y_test] = dataset.getTrainTestTensors(train, test);
                auto states = dataset.getStates(); // Get the states of the features Once they are discretized
                if (generate_fold_files)
                    generate_files(fileName, discretized, stratified, seed, nfold, X_train, y_train, X_test, y_test, train, test);
                auto progress = [this, nfold](char phase, bayesnet::status_t status) {
                    if (!quiet)
                        showProgress(nfold + 1, getColor(status), std::string(1, phase));
                    };
                auto outcome = run_fold(options, hyperparameters, fileName, X_train, y_train, X_test, y_test, features, className, states, progress);
                outcome.seed = seed;
                outcome.fold = nfold;
                setModelVersion(outcome.version);
                seed_score_sum += outcome.score_test;
                if (!quiet)
                    std::cout << "\b\b\b, " << std::flush;
                outcomes.push_back(std::move(outcome));
            }
            if (!quiet) {
                seed_timer.stop();
                std::cout << "end. " << std::setw(10) << std::right << seed_timer.getDurationString();
                if (randomSeeds.size() > 1) {
                    const auto seed_score_mean = seed_score_sum / static_cast<double>(nfolds);
                    std::cout << " " << Colors::YELLOW() << std::setw(9) << std::right << std::fixed << std::setprecision(7) << seed_score_mean << Colors::RESET();
                }
            }
            delete fold;
        }
        auto partial_result = build_partial_result(outcomes, options, fileName, n_samples, n_features, num_classes, hyperparameters.get(fileName));
        // Show Results
        if (!quiet)
            std::cout << " " << Colors::GREEN() << std::setw(9) << std::right << std::fixed << std::setprecision(7) << partial_result.getJson()["score"].get<double>() << Colors::RESET();
        addResult(partial_result);
    }
}