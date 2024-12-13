#include <iostream>
#include <argparse/argparse.hpp>
#include "main/Models.h"
#include "main/modelRegister.h"
#include "common/Paths.h"
#include "common/Colors.h"
#include "best/BestResults.h"
#include "config_platform.h"

void manageArguments(argparse::ArgumentParser& program)
{
    program.add_argument("-m", "--model")
        .help("Model to use or any")
        .default_value("any");
    program.add_argument("-d", "--dataset").default_value("any").help("Filter results of the selected model) (any for all datasets)");
    program.add_argument("-s", "--score").default_value("accuracy").help("Filter results of the score name supplied");
    program.add_argument("--friedman").help("Friedman test").default_value(false).implicit_value(true);
    program.add_argument("--excel").help("Output to excel").default_value(false).implicit_value(true);
    program.add_argument("--tex").help("Output result table to TeX file").default_value(false).implicit_value(true);
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
    argparse::ArgumentParser program("b_best", { platform_project_version.begin(), platform_project_version.end() });
    manageArguments(program);
    std::string model, dataset, score;
    bool build, report, friedman, excel, tex;
    double level;
    try {
        program.parse_args(argc, argv);
        model = program.get<std::string>("model");
        dataset = program.get<std::string>("dataset");
        score = program.get<std::string>("score");
        friedman = program.get<bool>("friedman");
        excel = program.get<bool>("excel");
        tex = program.get<bool>("tex");
        level = program.get<double>("level");
        if (model == "" || score == "") {
            throw std::runtime_error("Model and score name must be supplied");
        }
        if (friedman && (model != "any" || dataset != "any")) {
            std::cerr << "Friedman test can only be used with all models and all the datasets" << std::endl;
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
    auto results = platform::BestResults(platform::Paths::results(), score, model, dataset, friedman, level);
    if (model == "any") {
        results.buildAll();
        results.reportAll(excel, tex);
    } else {
        std::string fileName = results.build();
        std::cout << Colors::GREEN() << fileName << " created!" << Colors::RESET() << std::endl;
        results.reportSingle(excel);
    }
    std::cout << Colors::RESET();
    return 0;
}
