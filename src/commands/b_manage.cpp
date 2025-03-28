#include <iostream>
#include <sys/ioctl.h>
#include <utility>
#include <unistd.h>
#include <argparse/argparse.hpp>
#include "manage/ManageScreen.h"
#include <signal.h>
#include "config_platform.h"

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

void openFile(const std::string& fileName)
{
    // #ifdef __APPLE__
    //     // macOS uses the "open" command
    //     std::string command = "open";
    // #elif defined(__linux__)
    //     // Linux typically uses "xdg-open"
    //     std::string command = "xdg-open";
    // #else
    //     // For other OSes, do nothing or handle differently
    //     std::cerr << "Unsupported platform." << std::endl;
    //     return;
    // #endif
    //     execlp(command.c_str(), command.c_str(), fileName.c_str(), NULL);
#ifdef __APPLE__
    const char* tool = "/usr/bin/open";
#elif defined(__linux__)
    const char* tool = "/usr/bin/xdg-open";
#else
    std::cerr << "Unsupported platform." << std::endl;
    return;
#endif

    // We'll build an argv array for execve:
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(tool));               // argv[0]
    argv.push_back(const_cast<char*>(fileName.c_str()));   // argv[1]
    argv.push_back(nullptr);

    // Make a new environment array, skipping BASH_FUNC_ variables
    std::vector<std::string> filteredEnv;
    for (char** env = environ; *env != nullptr; ++env) {
        // *env is a string like "NAME=VALUE"
        // We want to skip those starting with "BASH_FUNC_"
        if (strncmp(*env, "BASH_FUNC_", 10) == 0) {
            // skip it
            continue;
        }
        filteredEnv.push_back(*env);
    }

    // Convert filteredEnv into a char* array
    std::vector<char*> envp;
    for (auto& var : filteredEnv) {
        envp.push_back(const_cast<char*>(var.c_str()));
    }
    envp.push_back(nullptr);

    // Now call execve with the cleaned environment
    // NOTE: You may need a full path to the tool if it's not in PATH, or use which() logic
    // For now, let's assume "open" or "xdg-open" is found in the default PATH:
    execve(tool, argv.data(), envp.data());

    // If we reach here, execve failed
    perror("execve failed");
    // This would terminate your current process if it's not in a child
    // Usually you'd do something like:
    _exit(EXIT_FAILURE);
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
    auto fileName = manager->getExcelFileName();
    delete manager;
    if (!fileName.empty()) {
        std::cout << "Opening " << fileName << std::endl;
        openFile(fileName);
    }
    return 0;
}
