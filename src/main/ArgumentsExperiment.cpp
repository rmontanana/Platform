#include "common/Datasets.h"
#include "common/DotEnv.h"
#include "common/Paths.h"
#include "main/Models.h"
#include "main/modelRegister.h"
#include "ArgumentsExperiment.h"
namespace platform {
    ArgumentsExperiment::ArgumentsExperiment(argparse::ArgumentParser& program, experiment_t type) : arguments{ program }, type{ type }
    {
    }
    void ArgumentsExperiment::add_arguments()
    {
        auto env = platform::DotEnv();
        auto datasets = platform::Datasets(false, platform::Paths::datasets());
        auto& group = arguments.add_mutually_exclusive_group(true);
        group.add_argument("-d", "--dataset")
            .help("Dataset file name: " + datasets.toString())
            .default_value("all")
            .action([](const std::string& value) {
            auto datasets = platform::Datasets(false, platform::Paths::datasets());
            static std::vector<std::string> choices_datasets(datasets.getNames());
            choices_datasets.push_back("all");
            if (find(choices_datasets.begin(), choices_datasets.end(), value) != choices_datasets.end()) {
                return value;
            }
            throw std::runtime_error("Dataset must be one of: " + datasets.toString());
                }
            );
        group.add_argument("--datasets").nargs(1, 50).help("Datasets file names 1..50 separated by spaces").default_value(std::vector<std::string>());
        group.add_argument("--datasets-file").default_value("").help("Datasets file name. Mutually exclusive with dataset. This file should contain a list of datasets to test.");
        arguments.add_argument("--hyperparameters").default_value("{}").help("Hyperparameters passed to the model in Experiment");
        arguments.add_argument("--hyper-file").default_value("").help("Hyperparameters file name." \
            "Mutually exclusive with hyperparameters. This file should contain hyperparameters for each dataset in json format.");
        arguments.add_argument("--hyper-best").default_value(false).help("Use best results of the model as source of hyperparameters").implicit_value(true);
        arguments.add_argument("-m", "--model")
            .help("Model to use: " + platform::Models::instance()->toString())
            .action([](const std::string& value) {
            static const std::vector<std::string> choices = platform::Models::instance()->getNames();
            if (find(choices.begin(), choices.end(), value) != choices.end()) {
                return value;
            }
            throw std::runtime_error("Model must be one of " + platform::Models::instance()->toString());
                }
            );
        arguments.add_argument("--title").default_value("").help("Experiment title");
        arguments.add_argument("--discretize").help("Discretize input dataset").default_value((bool)stoi(env.get("discretize"))).implicit_value(true);
        auto valid_choices = env.valid_tokens("discretize_algo");
        auto& disc_arg = arguments.add_argument("--discretize-algo").help("Algorithm to use in discretization. Valid values: " + env.valid_values("discretize_algo")).default_value(env.get("discretize_algo"));
        for (auto choice : valid_choices) {
            disc_arg.choices(choice);
        }
        valid_choices = env.valid_tokens("smooth_strat");
        auto& smooth_arg = arguments.add_argument("--smooth-strat").help("Smooth strategy used in Bayes Network node initialization. Valid values: " + env.valid_values("smooth_strat")).default_value(env.get("smooth_strat"));
        for (auto choice : valid_choices) {
            smooth_arg.choices(choice);
        }
        auto& score_arg = arguments.add_argument("-s", "--score").help("Score to use. Valid values: " + env.valid_values("score")).default_value(env.get("score"));
        valid_choices = env.valid_tokens("score");
        for (auto choice : valid_choices) {
            score_arg.choices(choice);
        }
        arguments.add_argument("--no-train-score").help("Don't compute train score").default_value(false).implicit_value(true);
        arguments.add_argument("--quiet").help("Don't display detailed progress").default_value(false).implicit_value(true);
        arguments.add_argument("--save").help("Save result (always save even if a dataset is supplied)").default_value(false).implicit_value(true);
        arguments.add_argument("--stratified").help("If Stratified KFold is to be done").default_value((bool)stoi(env.get("stratified"))).implicit_value(true);
        arguments.add_argument("-f", "--folds").help("Number of folds").default_value(stoi(env.get("n_folds"))).scan<'i', int>().action([](const std::string& value) {
            try {
                auto k = stoi(value);
                if (k < 2) {
                    throw std::runtime_error("Number of folds must be greater than 1");
                }
                return k;
            }
            catch (const runtime_error& err) {
                throw std::runtime_error(err.what());
            }
            catch (...) {
                throw std::runtime_error("Number of folds must be an integer");
            }});
            auto seed_values = env.getSeeds();
            arguments.add_argument("--seeds").nargs(1, 10).help("Random seeds. Set to -1 to have pseudo random").scan<'i', int>().default_value(seed_values);
            if (type == experiment_t::NORMAL) {
                arguments.add_argument("--generate-fold-files").help("generate fold information in datasets_experiment folder").default_value(false).implicit_value(true);
                arguments.add_argument("--graph").help("generate graphviz dot files with the model").default_value(false).implicit_value(true);
            }
    }
    void ArgumentsExperiment::parse_args(int argc, char** argv)
    {
        try {
            arguments.parse_args(argc, argv);
        }
        catch (const exception& err) {
            cerr << err.what() << std::endl;
            cerr << arguments;
            exit(1);
        }
        parse();
    }

    void ArgumentsExperiment::parse()
    {
        try {
            file_name = arguments.get<std::string>("dataset");
            file_names = arguments.get<std::vector<std::string>>("datasets");
            datasets_file = arguments.get<std::string>("datasets-file");
            model_name = arguments.get<std::string>("model");
            discretize_dataset = arguments.get<bool>("discretize");
            discretize_algo = arguments.get<std::string>("discretize-algo");
            smooth_strat = arguments.get<std::string>("smooth-strat");
            stratified = arguments.get<bool>("stratified");
            quiet = arguments.get<bool>("quiet");
            n_folds = arguments.get<int>("folds");
            score = arguments.get<std::string>("score");
            seeds = arguments.get<std::vector<int>>("seeds");
            auto hyperparameters = arguments.get<std::string>("hyperparameters");
            hyperparameters_json = json::parse(hyperparameters);
            hyperparameters_file = arguments.get<std::string>("hyper-file");
            no_train_score = arguments.get<bool>("no-train-score");
            hyper_best = arguments.get<bool>("hyper-best");
            if (hyper_best) {
                // Build the best results file_name
                hyperparameters_file = platform::Paths::results() + platform::Paths::bestResultsFile(score, model_name);
                // ignore this parameter
                hyperparameters = "{}";
            } else {
                if (hyperparameters_file != "" && hyperparameters != "{}") {
                    throw runtime_error("hyperparameters and hyper_file are mutually exclusive");
                }
            }
            title = arguments.get<std::string>("title");
            if (title == "" && file_name == "all") {
                throw runtime_error("title is mandatory if all datasets are to be tested");
            }
            saveResults = arguments.get<bool>("save");
            if (type == experiment_t::NORMAL) {
                graph = arguments.get<bool>("graph");
                generate_fold_files = arguments.get<bool>("generate-fold-files");
            } else {
                graph = false;
                generate_fold_files = false;
            }
        }
        catch (const exception& err) {
            cerr << err.what() << std::endl;
            cerr << arguments;
            exit(1);
        }
        auto datasets = platform::Datasets(false, platform::Paths::datasets());
        if (datasets_file != "") {
            ifstream catalog(datasets_file);
            if (catalog.is_open()) {
                std::string line;
                while (getline(catalog, line)) {
                    if (line.empty() || line[0] == '#') {
                        continue;
                    }
                    if (!datasets.isDataset(line)) {
                        cerr << "Dataset " << line << " not found" << std::endl;
                        exit(1);
                    }
                    filesToTest.push_back(line);
                }
                catalog.close();
                saveResults = true;
                if (title == "") {
                    title = "Test " + to_string(filesToTest.size()) + " datasets (" + datasets_file + ") "\
                        + model_name + " " + to_string(n_folds) + " folds";
                }
            } else {
                throw std::invalid_argument("Unable to open catalog file. [" + datasets_file + "]");
            }
        } else {
            if (file_names.size() > 0) {
                for (auto file : file_names) {
                    if (!datasets.isDataset(file)) {
                        cerr << "Dataset " << file << " not found" << std::endl;
                        exit(1);
                    }
                }
                filesToTest = file_names;
                saveResults = true;
                if (title == "") {
                    title = "Test " + to_string(file_names.size()) + " datasets " + model_name + " " + to_string(n_folds) + " folds";
                }
            } else {
                if (file_name != "all") {
                    if (!datasets.isDataset(file_name)) {
                        cerr << "Dataset " << file_name << " not found" << std::endl;
                        exit(1);
                    }
                    if (title == "") {
                        title = "Test " + file_name + " " + model_name + " " + to_string(n_folds) + " folds";
                    }
                    filesToTest.push_back(file_name);
                } else {
                    filesToTest = datasets.getNames();
                    saveResults = true;
                }
            }
        }
        if (hyperparameters_file != "") {
            test_hyperparams = platform::HyperParameters(datasets.getNames(), hyperparameters_file, hyper_best);
        } else {
            test_hyperparams = platform::HyperParameters(datasets.getNames(), hyperparameters_json);
        }
    }
    Experiment& ArgumentsExperiment::initializedExperiment()
    {
        auto env = platform::DotEnv();
        experiment.setTitle(title).setLanguage("c++").setLanguageVersion("gcc 14.1.1");
        experiment.setDiscretizationAlgorithm(discretize_algo).setSmoothSrategy(smooth_strat);
        experiment.setDiscretized(discretize_dataset).setModel(model_name).setPlatform(env.get("platform"));
        experiment.setStratified(stratified).setNFolds(n_folds).setScoreName(score);
        experiment.setHyperparameters(test_hyperparams);
        for (auto seed : seeds) {
            experiment.addRandomSeed(seed);
        }
        experiment.setFilesToTest(filesToTest);
        experiment.setQuiet(quiet);
        experiment.setNoTrainScore(no_train_score);
        experiment.setGenerateFoldFiles(generate_fold_files);
        experiment.setGraph(graph);
        return experiment;
    }
}