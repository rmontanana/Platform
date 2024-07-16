#include <iostream>
#include <sys/ioctl.h>
#include <utility>
#include <unistd.h>
#include <argparse/argparse.hpp>
#include "manage/ManageScreen.h"
#include <signal.h>
#include "config.h"

platform::ManageScreen* manager = nullptr;

void manageArguments(argparse::ArgumentParser& program, int argc, char** argv)
{
    program.add_argument("-m", "--model").default_value("any").help("Filter results of the selected model)");
    program.add_argument("-s", "--score").default_value("any").help("Filter results of the score name supplied");
    program.add_argument("--platform").default_value("any").help("Filter results of the selected platform");
    program.add_argument("--complete").help("Show only results with all datasets").default_value(false).implicit_value(true);
    program.add_argument("--partial").help("Show only partial results").default_value(false).implicit_value(true);
    program.add_argument("--compare").help("Compare with best results").default_value(false).implicit_value(true);
    try {
        program.parse_args(argc, argv);
        auto platform = program.get<std::string>("platform");
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

std::pair<int, int> numRowsCols()
{
#ifdef TIOCGSIZE
    struct ttysize ts;
    ioctl(STDIN_FILENO, TIOCGSIZE, &ts);
    return { ts.ts_lines, ts.ts_cols };
#elif defined(TIOCGWINSZ)
    struct winsize ts;
    ioctl(STDIN_FILENO, TIOCGWINSZ, &ts);
    return { ts.ws_row, ts.ws_col };
#endif /* TIOCGSIZE */
}
void handleResize(int sig)
{
    auto [rows, cols] = numRowsCols();
    manager->updateSize(rows, cols);
}

int main(int argc, char** argv)
{
    auto program = argparse::ArgumentParser("b_manage", { platform_project_version.begin(), platform_project_version.end() });
    manageArguments(program, argc, argv);
    std::string model = program.get<std::string>("model");
    std::string score = program.get<std::string>("score");
    std::string platform = program.get<std::string>("platform");
    bool complete = program.get<bool>("complete");
    bool partial = program.get<bool>("partial");
    bool compare = program.get<bool>("compare");
    if (complete)
        partial = false;
    signal(SIGWINCH, handleResize);
    auto [rows, cols] = numRowsCols();
    manager = new platform::ManageScreen(rows, cols, model, score, platform, complete, partial, compare);
    manager->doMenu();
    delete manager;
    return 0;
}
