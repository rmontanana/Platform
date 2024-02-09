#include <fstream>
#include "Experiment.h"
#include "Datasets.h"
#include "Models.h"
#include "ReportConsole.h"
#include "Paths.h"
namespace platform {
    using json = nlohmann::json;
    std::string get_date()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%Y-%m-%d");
        return oss.str();
    }
    std::string get_time()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%H:%M:%S");
        return oss.str();
    }
    std::string Experiment::get_file_name()
    {
        std::string result = "results_" + score_name + "_" + model + "_" + platform + "_" + get_date() + "_" + get_time() + "_" + (stratified ? "1" : "0") + ".json";
        return result;
    }

    json Experiment::build_json()
    {
        json result;
        result["title"] = title;
        result["date"] = get_date();
        result["time"] = get_time();
        result["model"] = model;
        result["version"] = model_version;
        result["platform"] = platform;
        result["score_name"] = score_name;
        result["language"] = language;
        result["language_version"] = language_version;
        result["discretized"] = discretized;
        result["stratified"] = stratified;
        result["folds"] = nfolds;
        result["seeds"] = randomSeeds;
        result["duration"] = duration;
        result["results"] = json::array();
        for (const auto& r : results) {
            json j;
            j["dataset"] = r.getDataset();
            j["hyperparameters"] = r.getHyperparameters();
            j["samples"] = r.getSamples();
            j["features"] = r.getFeatures();
            j["classes"] = r.getClasses();
            j["score_train"] = r.getScoreTrain();
            j["score_test"] = r.getScoreTest();
            j["score"] = r.getScoreTest();
            j["score_std"] = r.getScoreTestStd();
            j["score_train_std"] = r.getScoreTrainStd();
            j["score_test_std"] = r.getScoreTestStd();
            j["train_time"] = r.getTrainTime();
            j["train_time_std"] = r.getTrainTimeStd();
            j["test_time"] = r.getTestTime();
            j["test_time_std"] = r.getTestTimeStd();
            j["time"] = r.getTestTime() + r.getTrainTime();
            j["time_std"] = r.getTestTimeStd() + r.getTrainTimeStd();
            j["scores_train"] = r.getScoresTrain();
            j["scores_test"] = r.getScoresTest();
            j["times_train"] = r.getTimesTrain();
            j["times_test"] = r.getTimesTest();
            j["nodes"] = r.getNodes();
            j["leaves"] = r.getLeaves();
            j["depth"] = r.getDepth();
            j["notes"] = r.getNotes();
            result["results"].push_back(j);
        }
        return result;
    }
    void Experiment::save(const std::string& path)
    {
        json data = build_json();
        ofstream file(path + "/" + get_file_name());
        file << data;
        file.close();
    }

    void Experiment::report()
    {
        json data = build_json();
        ReportConsole report(data);
        report.show();
    }

    void Experiment::show()
    {
        json data = build_json();
        std::cout << data.dump(4) << std::endl;
    }

    void Experiment::go(std::vector<std::string> filesToProcess, bool quiet, bool no_train_score)
    {
        for (auto fileName : filesToProcess) {
            if (fileName.size() > max_name)
                max_name = fileName.size();
        }
        std::cout << Colors::MAGENTA() << "*** Starting experiment: " << title << " ***" << Colors::RESET() << std::endl << std::endl;
        if (!quiet) {
            std::cout << Colors::GREEN() << " Status Meaning" << std::endl;
            std::cout << " ------ -----------------------------" << Colors::RESET() << std::endl;
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
        if (!quiet) {
            std::cout << " " << setw(5) << samples << " " << setw(5) << features.size() << flush;
        }
        // Prepare Resu lt
        auto result = Result();
        auto [values, counts] = at::_unique(y);
        result.setSamples(X.size(1)).setFeatures(X.size(0)).setClasses(values.size(0));
        result.setHyperparameters(hyperparameters.get(fileName));
        // Initialize results std::vectors
        int nResults = nfolds * static_cast<int>(randomSeeds.size());
        auto accuracy_test = torch::zeros({ nResults }, torch::kFloat64);
        auto accuracy_train = torch::zeros({ nResults }, torch::kFloat64);
        auto train_time = torch::zeros({ nResults }, torch::kFloat64);
        auto test_time = torch::zeros({ nResults }, torch::kFloat64);
        auto nodes = torch::zeros({ nResults }, torch::kFloat64);
        auto edges = torch::zeros({ nResults }, torch::kFloat64);
        auto num_states = torch::zeros({ nResults }, torch::kFloat64);
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
                auto clf = Models::instance()->create(model);
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
                notes.insert(notes.end(), clf_notes.begin(), clf_notes.end());
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
                auto accuracy_test_value = clf->score(X_test, y_test);
                test_time[item] = test_timer.getDuration();
                accuracy_train[item] = accuracy_train_value;
                accuracy_test[item] = accuracy_test_value;
                if (!quiet)
                    std::cout << "\b\b\b, " << flush;
                // Store results and times in std::vector
                result.addScoreTrain(accuracy_train_value);
                result.addScoreTest(accuracy_test_value);
                result.addTimeTrain(train_time[item].item<double>());
                result.addTimeTest(test_time[item].item<double>());
                item++;
            }
            if (!quiet)
                std::cout << "end. " << flush;
            delete fold;
        }
        result.setScoreTest(torch::mean(accuracy_test).item<double>()).setScoreTrain(torch::mean(accuracy_train).item<double>());
        result.setScoreTestStd(torch::std(accuracy_test).item<double>()).setScoreTrainStd(torch::std(accuracy_train).item<double>());
        result.setTrainTime(torch::mean(train_time).item<double>()).setTestTime(torch::mean(test_time).item<double>());
        result.setTestTimeStd(torch::std(test_time).item<double>()).setTrainTimeStd(torch::std(train_time).item<double>());
        result.setNodes(torch::mean(nodes).item<double>()).setLeaves(torch::mean(edges).item<double>()).setDepth(torch::mean(num_states).item<double>());
        result.setDataset(fileName).setNotes(notes);
        addResult(result);
    }
}