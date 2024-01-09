#include <iostream>
#include <argparse/argparse.hpp>
#include <map>
#include <nlohmann/json.hpp>
#include <mpi.h>
#include "DotEnv.h"
#include "Models.h"
#include "modelRegister.h"
#include "GridSearch.h"
#include "Paths.h"
#include "Timer.h"
#include "Colors.h"
#include "config.h"

using json = nlohmann::json;
const int MAXL = 133;

void manageArguments(argparse::ArgumentParser& program)
{
    auto env = platform::DotEnv();
    auto& group = program.add_mutually_exclusive_group(true);
    program.add_argument("-m", "--model")
        .help("Model to use " + platform::Models::instance()->tostring())
        .action([](const std::string& value) {
        static const std::vector<std::string> choices = platform::Models::instance()->getNames();
        if (find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw std::runtime_error("Model must be one of " + platform::Models::instance()->tostring());
            }
    );
    group.add_argument("--dump").help("Show the grid combinations").default_value(false).implicit_value(true);
    group.add_argument("--report").help("Report the computed hyperparameters").default_value(false).implicit_value(true);
    group.add_argument("--compute").help("Perform computation of the grid output hyperparameters").default_value(false).implicit_value(true);
    program.add_argument("--discretize").help("Discretize input datasets").default_value((bool)stoi(env.get("discretize"))).implicit_value(true);
    program.add_argument("--stratified").help("If Stratified KFold is to be done").default_value((bool)stoi(env.get("stratified"))).implicit_value(true);
    program.add_argument("--quiet").help("Don't display detailed progress").default_value(false).implicit_value(true);
    program.add_argument("--continue").help("Continue computing from that dataset").default_value(platform::GridSearch::NO_CONTINUE());
    program.add_argument("--only").help("Used with continue to compute that dataset only").default_value(false).implicit_value(true);
    program.add_argument("--exclude").default_value("[]").help("Datasets to exclude in json format, e.g. [\"dataset1\", \"dataset2\"]");
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

void list_dump(std::string& model)
{
    auto data = platform::GridData(platform::Paths::grid_input(model));
    std::cout << Colors::MAGENTA() << "Listing configuration input file (Grid)" << std::endl << std::endl;
    int index = 0;
    int max_hyper = 15;
    int max_dataset = 7;
    auto combinations = data.getGridFile();
    for (auto const& item : combinations) {
        if (item.first.size() > max_dataset) {
            max_dataset = item.first.size();
        }
        if (item.second.dump().size() > max_hyper) {
            max_hyper = item.second.dump().size();
        }
    }
    std::cout << Colors::GREEN() << left << " #  " << left << setw(max_dataset) << "Dataset" << " #Com. "
        << setw(max_hyper) << "Hyperparameters" << std::endl;
    std::cout << "=== " << string(max_dataset, '=') << " ===== " << string(max_hyper, '=') << std::endl;
    bool odd = true;
    for (auto const& item : combinations) {
        auto color = odd ? Colors::CYAN() : Colors::BLUE();
        std::cout << color;
        auto num_combinations = data.getNumCombinations(item.first);
        std::cout << setw(3) << fixed << right << ++index << left << " " << setw(max_dataset) << item.first
            << " " << setw(5) << right << num_combinations << " " << setw(max_hyper) << item.second.dump() << std::endl;
        odd = !odd;
    }
    std::cout << Colors::RESET() << std::endl;
}
std::string headerLine(const std::string& text, int utf = 0)
{
    int n = MAXL - text.length() - 3;
    n = n < 0 ? 0 : n;
    return "* " + text + std::string(n + utf, ' ') + "*\n";
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
    bool odd = true;
    int index = 0;
    for (const auto& item : results["results"].items()) {
        auto color = odd ? Colors::CYAN() : Colors::BLUE();
        auto value = item.value();
        std::cout << color;
        std::cout << std::setw(3) << std::right << index++ << " ";
        std::cout << left << setw(spaces) << item.key() << " " << value["date"].get<string>()
            << " " << setw(8) << right << value["duration"].get<string>() << " " << setw(8) << setprecision(6)
            << fixed << right << value["score"].get<double>() << " " << value["hyperparameters"].dump() << std::endl;
        odd = !odd;
    }
    std::cout << Colors::RESET() << std::endl;
}

/*
 * Main
 */
int main(int argc, char** argv)
{
    argparse::ArgumentParser program("b_grid", { project_version.begin(), project_version.end() });
    manageArguments(program);
    struct platform::ConfigGrid config;
    bool dump, compute;
    try {
        program.parse_args(argc, argv);
        config.model = program.get<std::string>("model");
        config.score = program.get<std::string>("score");
        config.discretize = program.get<bool>("discretize");
        config.stratified = program.get<bool>("stratified");
        config.n_folds = program.get<int>("folds");
        config.quiet = program.get<bool>("quiet");
        config.only = program.get<bool>("only");
        config.seeds = program.get<std::vector<int>>("seeds");
        config.nested = program.get<int>("nested");
        config.continue_from = program.get<std::string>("continue");
        if (config.continue_from == platform::GridSearch::NO_CONTINUE() && config.only) {
            throw std::runtime_error("Cannot use --only without --continue");
        }
        dump = program.get<bool>("dump");
        compute = program.get<bool>("compute");
        if (dump && (config.continue_from != platform::GridSearch::NO_CONTINUE() || config.only)) {
            throw std::runtime_error("Cannot use --dump with --continue or --only");
        }
        auto excluded = program.get<std::string>("exclude");
        config.excluded = json::parse(excluded);
    }
    catch (const exception& err) {
        cerr << err.what() << std::endl;
        cerr << program;
        exit(1);
    }
    /*
     * Begin Processing
     */
    auto env = platform::DotEnv();
    config.platform = env.get("platform");
    platform::Paths::createPath(platform::Paths::grid());
    auto grid_search = platform::GridSearch(config);
    platform::Timer timer;
    timer.start();
    if (dump) {
        list_dump(config.model);
    } else {
        if (compute) {
            struct platform::ConfigMPI mpi_config;
            mpi_config.manager = 0; // which process is the manager
            MPI_Init(&argc, &argv);
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
        } else {
            // List results
            auto results = grid_search.loadResults();
            if (results.empty()) {
                std::cout << "** No results found" << std::endl;
            } else {
                list_results(results, config.model);
            }
        }
    }
    std::cout << "Done!" << std::endl;
    return 0;
}
