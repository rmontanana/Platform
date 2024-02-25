#include <iostream>
#include <argparse/argparse.hpp>
#include "ManageResults.h"
#include "config.h"


void manageArguments(argparse::ArgumentParser& program, int argc, char** argv)
{
    program.add_argument("-n", "--number").default_value(0).help("Number of results to show (0 = all)").scan<'i', int>();
    program.add_argument("-m", "--model").default_value("any").help("Filter results of the selected model)");
    program.add_argument("-s", "--score").default_value("any").help("Filter results of the score name supplied");
    program.add_argument("--complete").help("Show only results with all datasets").default_value(false).implicit_value(true);
    program.add_argument("--partial").help("Show only partial results").default_value(false).implicit_value(true);
    program.add_argument("--compare").help("Compare with best results").default_value(false).implicit_value(true);
    try {
        program.parse_args(argc, argv);
        auto number = program.get<int>("number");
        if (number < 0) {
            throw std::runtime_error("Number of results must be greater than or equal to 0");
        }
        auto model = program.get<std::string>("model");
        auto score = program.get<std::string>("score");
        auto complete = program.get<bool>("complete");
        auto partial = program.get<bool>("partial");
        auto compare = program.get<bool>("compare");
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        exit(1);
    }
}

int main(int argc, char** argv)
{
    auto program = argparse::ArgumentParser("b_manage", { project_version.begin(), project_version.end() });
    manageArguments(program, argc, argv);
    int number = program.get<int>("number");
    std::string model = program.get<std::string>("model");
    std::string score = program.get<std::string>("score");
    auto complete = program.get<bool>("complete");
    auto partial = program.get<bool>("partial");
    auto compare = program.get<bool>("compare");
    if (complete)
        partial = false;
    auto manager = platform::ManageResults(number, model, score, complete, partial, compare);
    manager.doMenu();
    return 0;
}
