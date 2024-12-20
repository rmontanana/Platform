#include <iostream>
#include <argparse/argparse.hpp>
#include <map>
#include <tuple>
#include <nlohmann/json.hpp>
#include <mpi.h>
#include "main/Models.h"
#include "main/modelRegister.h"
#include "common/Paths.h"
#include "common/Timer.h"
#include "common/Colors.h"
#include "common/DotEnv.h"
#include "grid/GridSearch.h"
#include "grid/GridExperiment.h"
#include "config_platform.h"

using json = nlohmann::ordered_json;
const int MAXL = 133;

void assignModel(argparse::ArgumentParser& parser)
{
    auto models = platform::Models::instance();
    parser.add_argument("-m", "--model")
        .help("Model to use " + models->toString())
        .required()
        .action([models](const std::string& value) {
        static const std::vector<std::string> choices = models->getNames();
        if (find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw std::runtime_error("Model must be one of " + models->toString());
            }
        );
}
void add_experiment_args(argparse::ArgumentParser& program)
{
    auto env = platform::DotEnv();
    auto datasets = platform::Datasets(false, platform::Paths::datasets());
    auto& group = program.add_mutually_exclusive_group(true);
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
    program.add_argument("--hyperparameters").default_value("{}").help("Hyperparameters passed to the model in Experiment");
    program.add_argument("--hyper-file").default_value("").help("Hyperparameters file name." \
        "Mutually exclusive with hyperparameters. This file should contain hyperparameters for each dataset in json format.");
    program.add_argument("--hyper-best").default_value(false).help("Use best results of the model as source of hyperparameters").implicit_value(true);
    program.add_argument("-m", "--model")
        .help("Model to use: " + platform::Models::instance()->toString())
        .action([](const std::string& value) {
        static const std::vector<std::string> choices = platform::Models::instance()->getNames();
        if (find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw std::runtime_error("Model must be one of " + platform::Models::instance()->toString());
            }
        );
    program.add_argument("--title").default_value("").help("Experiment title");
    program.add_argument("--discretize").help("Discretize input dataset").default_value((bool)stoi(env.get("discretize"))).implicit_value(true);
    auto valid_choices = env.valid_tokens("discretize_algo");
    auto& disc_arg = program.add_argument("--discretize-algo").help("Algorithm to use in discretization. Valid values: " + env.valid_values("discretize_algo")).default_value(env.get("discretize_algo"));
    for (auto choice : valid_choices) {
        disc_arg.choices(choice);
    }
    valid_choices = env.valid_tokens("smooth_strat");
    auto& smooth_arg = program.add_argument("--smooth-strat").help("Smooth strategy used in Bayes Network node initialization. Valid values: " + env.valid_values("smooth_strat")).default_value(env.get("smooth_strat"));
    for (auto choice : valid_choices) {
        smooth_arg.choices(choice);
    }
    auto& score_arg = program.add_argument("-s", "--score").help("Score to use. Valid values: " + env.valid_values("score")).default_value(env.get("score"));
    valid_choices = env.valid_tokens("score");
    for (auto choice : valid_choices) {
        score_arg.choices(choice);
    }
    program.add_argument("--generate-fold-files").help("generate fold information in datasets_experiment folder").default_value(false).implicit_value(true);
    program.add_argument("--graph").help("generate graphviz dot files with the model").default_value(false).implicit_value(true);
    program.add_argument("--no-train-score").help("Don't compute train score").default_value(false).implicit_value(true);
    program.add_argument("--quiet").help("Don't display detailed progress").default_value(false).implicit_value(true);
    program.add_argument("--save").help("Save result (always save if no dataset is supplied)").default_value(false).implicit_value(true);
    program.add_argument("--stratified").help("If Stratified KFold is to be done").default_value((bool)stoi(env.get("stratified"))).implicit_value(true);
    program.add_argument("-f", "--folds").help("Number of folds").default_value(stoi(env.get("n_folds"))).scan<'i', int>().action([](const std::string& value) {
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
        program.add_argument("--seeds").nargs(1, 10).help("Random seeds. Set to -1 to have pseudo random").scan<'i', int>().default_value(seed_values);
}
void add_compute_args(argparse::ArgumentParser& program)
{
    auto env = platform::DotEnv();
    program.add_argument("--discretize").help("Discretize input datasets").default_value((bool)stoi(env.get("discretize"))).implicit_value(true);
    program.add_argument("--stratified").help("If Stratified KFold is to be done").default_value((bool)stoi(env.get("stratified"))).implicit_value(true);
    program.add_argument("--quiet").help("Don't display detailed progress").default_value(false).implicit_value(true);
    program.add_argument("--continue").help("Continue computing from that dataset").default_value(platform::GridSearch::NO_CONTINUE());
    program.add_argument("--only").help("Used with continue to compute that dataset only").default_value(false).implicit_value(true);
    program.add_argument("--exclude").default_value("[]").help("Datasets to exclude in json format, e.g. [\"dataset1\", \"dataset2\"]");
    auto valid_choices = env.valid_tokens("smooth_strat");
    auto& smooth_arg = program.add_argument("--smooth-strat").help("Smooth strategy used in Bayes Network node initialization. Valid values: " + env.valid_values("smooth_strat")).default_value(env.get("smooth_strat"));
    for (auto choice : valid_choices) {
        smooth_arg.choices(choice);
    }
    program.add_argument("--nested").help("Set the double/nested cross validation number of folds").default_value(5).scan<'i', int>().action([](const std::string& value) {
        try {
            auto k = stoi(value);
            if (k < 2) {
                throw std::runtime_error("Number of nested folds must be greater than 1");
            }
            return k;
        }
        catch (const runtime_error& err) {
            throw std::runtime_error(err.what());
        }
        catch (...) {
            throw std::runtime_error("Number of nested folds must be an integer");
        }});
        program.add_argument("--score").help("Score used in gridsearch").default_value("accuracy");
        program.add_argument("-f", "--folds").help("Number of folds").default_value(stoi(env.get("n_folds"))).scan<'i', int>().action([](const std::string& value) {
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
            program.add_argument("-s", "--seeds").nargs(1, 10).help("Random seeds. Set to -1 to have pseudo random").scan<'i', int>().default_value(seed_values);
}
std::string headerLine(const std::string& text, int utf = 0)
{
    int n = MAXL - text.length() - 3;
    n = n < 0 ? 0 : n;
    return "* " + text + std::string(n + utf, ' ') + "*\n";
}
void list_dump(std::string& model)
{
    auto data = platform::GridData(platform::Paths::grid_input(model));
    std::cout << Colors::MAGENTA() << std::string(MAXL, '*') << std::endl;
    std::cout << headerLine("Listing configuration input file (Grid)");
    std::cout << headerLine("Model: " + model);
    std::cout << Colors::MAGENTA() << std::string(MAXL, '*') << std::endl;
    int index = 0;
    int max_hyper = 15;
    int max_dataset = 7;
    auto combinations = data.getGridFile();
    for (auto const& item : combinations) {
        if (item.first.size() > max_dataset) {
            max_dataset = item.first.size();
        }
        for (auto const& [key, value] : item.second.items()) {
            if (value.dump().size() > max_hyper) {
                max_hyper = value.dump().size();
            }
        }
    }
    std::cout << Colors::GREEN() << left << " #  " << left << setw(max_dataset) << "Dataset" << " #Com. "
        << setw(max_hyper) << "Hyperparameters" << std::endl;
    std::cout << "=== " << string(max_dataset, '=') << " ===== " << string(max_hyper, '=') << std::endl;
    int i = 0;
    for (auto const& item : combinations) {
        auto color = (i++ % 2) ? Colors::CYAN() : Colors::BLUE();
        std::cout << color;
        auto num_combinations = data.getNumCombinations(item.first);
        std::cout << setw(3) << fixed << right << ++index << left << " " << setw(max_dataset) << item.first
            << " " << setw(5) << right << num_combinations << " ";
        std::string prefix = "";
        for (auto const& [key, value] : item.second.items()) {
            std::cout << prefix << setw(max_hyper) << std::left << value.dump() << std::endl;
            prefix = string(11 + max_dataset, ' ');
        }
    }
    std::cout << Colors::RESET() << std::endl;
}
void list_results(json& results, std::string& model)
{
    std::cout << Colors::MAGENTA() << std::string(MAXL, '*') << std::endl;
    std::cout << headerLine("Listing computed hyperparameters for model " + model);
    std::cout << headerLine("Date & time: " + results["date"].get<std::string>() + " Duration: " + results["duration"].get<std::string>());
    std::cout << headerLine("Score: " + results["score"].get<std::string>());
    std::cout << headerLine(
        "Random seeds: " + results["seeds"].dump()
        + " Discretized: " + (results["discretize"].get<bool>() ? "True" : "False")
        + " Stratified: " + (results["stratified"].get<bool>() ? "True" : "False")
        + " #Folds: " + std::to_string(results["n_folds"].get<int>())
        + " Nested: " + (results["nested"].get<int>() == 0 ? "False" : to_string(results["nested"].get<int>()))
    );
    std::cout << std::string(MAXL, '*') << std::endl;
    int spaces = 7;
    int hyperparameters_spaces = 15;
    for (const auto& item : results["results"].items()) {
        auto key = item.key();
        auto value = item.value();
        if (key.size() > spaces) {
            spaces = key.size();
        }
        if (value["hyperparameters"].dump().size() > hyperparameters_spaces) {
            hyperparameters_spaces = value["hyperparameters"].dump().size();
        }
    }
    std::cout << Colors::GREEN() << " #  " << left << setw(spaces) << "Dataset" << " " << setw(19) << "Date" << " "
        << "Duration " << setw(8) << "Score" << " " << "Hyperparameters" << std::endl;
    std::cout << "=== " << string(spaces, '=') << " " << string(19, '=') << " " << string(8, '=') << " "
        << string(8, '=') << " " << string(hyperparameters_spaces, '=') << std::endl;
    int index = 0;
    for (const auto& item : results["results"].items()) {
        auto color = (index % 2) ? Colors::CYAN() : Colors::BLUE();
        auto value = item.value();
        std::cout << color;
        std::cout << std::setw(3) << std::right << index++ << " ";
        std::cout << left << setw(spaces) << item.key() << " " << value["date"].get<string>()
            << " " << setw(8) << right << value["duration"].get<string>() << " " << setw(8) << setprecision(6)
            << fixed << right << value["score"].get<double>() << " " << value["hyperparameters"].dump() << std::endl;
    }
    std::cout << Colors::RESET() << std::endl;
}

/*
 * Main
 */
void dump(argparse::ArgumentParser& program)
{
    auto model = program.get<std::string>("model");
    list_dump(model);
}
void report(argparse::ArgumentParser& program)
{
    // List results
    struct platform::ConfigGrid config;
    config.model = program.get<std::string>("model");
    auto grid_search = platform::GridSearch(config);
    auto results = grid_search.loadResults();
    if (results.empty()) {
        std::cout << "** No results found" << std::endl;
    } else {
        list_results(results, config.model);
    }
}
void compute(argparse::ArgumentParser& program)
{
    struct platform::ConfigGrid config;
    config.model = program.get<std::string>("model");
    config.score = program.get<std::string>("score");
    config.discretize = program.get<bool>("discretize");
    config.stratified = program.get<bool>("stratified");
    config.smooth_strategy = program.get<std::string>("smooth-strat");
    config.n_folds = program.get<int>("folds");
    config.quiet = program.get<bool>("quiet");
    config.only = program.get<bool>("only");
    config.seeds = program.get<std::vector<int>>("seeds");
    config.nested = program.get<int>("nested");
    config.continue_from = program.get<std::string>("continue");
    if (config.continue_from == platform::GridSearch::NO_CONTINUE() && config.only) {
        throw std::runtime_error("Cannot use --only without --continue");
    }
    auto excluded = program.get<std::string>("exclude");
    config.excluded = json::parse(excluded);

    auto env = platform::DotEnv();
    config.platform = env.get("platform");
    platform::Paths::createPath(platform::Paths::grid());
    auto grid_search = platform::GridSearch(config);
    platform::Timer timer;
    timer.start();
    struct platform::ConfigMPI mpi_config;
    mpi_config.manager = 0; // which process is the manager
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_config.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_config.n_procs);
    if (mpi_config.n_procs < 2) {
        throw std::runtime_error("Cannot use --compute with less than 2 mpi processes, try mpirun -np 2 ...");
    }
    grid_search.go(mpi_config);
    if (mpi_config.rank == mpi_config.manager) {
        auto results = grid_search.loadResults();
        list_results(results, config.model);
        std::cout << "Process took " << timer.getDurationString() << std::endl;
    }
    MPI_Finalize();
}
void experiment(argparse::ArgumentParser& program)
{
    struct platform::ConfigGrid config;
    config.model = program.get<std::string>("model");
    config.score = program.get<std::string>("score");
    config.discretize = program.get<bool>("discretize");
    config.stratified = program.get<bool>("stratified");
    config.smooth_strategy = program.get<std::string>("smooth-strat");
    config.n_folds = program.get<int>("folds");
    config.quiet = program.get<bool>("quiet");
    config.seeds = program.get<std::vector<int>>("seeds");

    auto env = platform::DotEnv();
    config.platform = env.get("platform");
    platform::Paths::createPath(platform::Paths::grid());
    // auto grid_experiment = platform::GridExperiment(config);
    platform::Timer timer;
    timer.start();
    struct platform::ConfigMPI mpi_config;
    mpi_config.manager = 0; // which process is the manager
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_config.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_config.n_procs);
    if (mpi_config.n_procs < 2) {
        throw std::runtime_error("Cannot use --compute with less than 2 mpi processes, try mpirun -np 2 ...");
    }
    // grid_experiment.go(mpi_config);
    if (mpi_config.rank == mpi_config.manager) {
        // auto results = grid_experiment.loadResults();
        // list_results(results, config.model);
        std::cout << "Process took " << timer.getDurationString() << std::endl;
    }
    MPI_Finalize();
}
int main(int argc, char** argv)
{
    //
    // Manage arguments
    //
    argparse::ArgumentParser program("b_grid", { platform_project_version.begin(), platform_project_version.end() });
    // grid dump subparser
    argparse::ArgumentParser dump_command("dump");
    dump_command.add_description("Dump the combinations of hyperparameters of a model.");
    assignModel(dump_command);

    // grid report subparser
    argparse::ArgumentParser report_command("report");
    assignModel(report_command);
    report_command.add_description("Report the computed hyperparameters of a model.");

    // grid compute subparser
    argparse::ArgumentParser compute_command("compute");
    compute_command.add_description("Compute using mpi the hyperparameters of a model.");
    assignModel(compute_command);
    add_compute_args(compute_command);

    // grid experiment subparser
    argparse::ArgumentParser experiment_command("experiment");
    experiment_command.add_description("Experiment like b_main using mpi.");
    assignModel(experiment_command);
    add_experiment_args(experiment_command);

    program.add_subparser(dump_command);
    program.add_subparser(report_command);
    program.add_subparser(compute_command);
    program.add_subparser(experiment_command);

    // 
    // Process options
    //
    try {
        program.parse_args(argc, argv);
        bool found = false;
        map<std::string, void(*)(argparse::ArgumentParser&)> commands = { {"dump", &dump}, {"report", &report}, {"compute", &compute}, { "experiment",&experiment } };
        for (const auto& command : commands) {
            if (program.is_subcommand_used(command.first)) {
                std::invoke(command.second, program.at<argparse::ArgumentParser>(command.first));
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("You must specify one of the following commands: dump, report, compute\n");
        }
    }
    catch (const exception& err) {
        cerr << err.what() << std::endl;
        cerr << program;
        exit(1);
    }
    std::cout << "Done!" << std::endl;
    return 0;
}
