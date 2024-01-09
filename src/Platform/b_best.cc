#include <iostream>
#include <argparse/argparse.hpp>
#include "Paths.h"
#include "BestResults.h"
#include "Colors.h"
#include "config.h"

void manageArguments(argparse::ArgumentParser& program, int argc, char** argv)
{
    program.add_argument("-m", "--model").default_value("").help("Filter results of the selected model) (any for all models)");
    program.add_argument("-s", "--score").default_value("").help("Filter results of the score name supplied");
    program.add_argument("--build").help("build best score results file").default_value(false).implicit_value(true);
    program.add_argument("--report").help("report of best score results file").default_value(false).implicit_value(true);
    program.add_argument("--friedman").help("Friedman test").default_value(false).implicit_value(true);
    program.add_argument("--excel").help("Output to excel").default_value(false).implicit_value(true);
    program.add_argument("--level").help("significance level").default_value(0.05).scan<'g', double>().action([](const std::string& value) {
        try {
            auto k = std::stod(value);
            if (k < 0.01 || k > 0.15) {
                throw std::runtime_error("Significance level hast to be a number in [0.01, 0.15]");
            }
            return k;
        }
        catch (const std::runtime_error& err) {
            throw std::runtime_error(err.what());
        }
        catch (...) {
            throw std::runtime_error("Number of folds must be an decimal number");
        }});
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("b_best", { project_version.begin(), project_version.end() });
    manageArguments(program, argc, argv);
    std::string model, score;
    bool build, report, friedman, excel;
    double level;
    try {
        program.parse_args(argc, argv);
        model = program.get<std::string>("model");
        score = program.get<std::string>("score");
        build = program.get<bool>("build");
        report = program.get<bool>("report");
        friedman = program.get<bool>("friedman");
        excel = program.get<bool>("excel");
        level = program.get<double>("level");
        if (model == "" || score == "") {
            throw std::runtime_error("Model and score name must be supplied");
        }
        if (friedman && model != "any") {
            std::cerr << "Friedman test can only be used with all models" << std::endl;
            std::cerr << program;
            exit(1);
        }
        if (!report && !build) {
            std::cerr << "Either build, report or both, have to be selected to do anything!" << std::endl;
            std::cerr << program;
            exit(1);
        }
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        exit(1);
    }
    // Generate report
    auto results = platform::BestResults(platform::Paths::results(), score, model, friedman, level);
    if (build) {
        if (model == "any") {
            results.buildAll();
        } else {
            std::string fileName = results.build();
            std::cout << Colors::GREEN() << fileName << " created!" << Colors::RESET() << std::endl;
        }
    }
    if (report) {
        if (model == "any") {
            results.reportAll(excel);
        } else {
            results.reportSingle(excel);
        }
    }
    return 0;
}
