#ifndef PATHS_H
#define PATHS_H
#include <string>
#include <filesystem>
#include "DotEnv.h"
namespace platform {
    class Paths {
    public:
        static std::string createIfNotExists(const std::string& folder)
        {
            if (!std::filesystem::exists(folder)) {
                std::filesystem::create_directory(folder);
            }
            return folder;
        }
        static std::string results() { return createIfNotExists("results/"); }
        static std::string hiddenResults() { return createIfNotExists("hidden_results/"); }
        static std::string excel() { return createIfNotExists("excel/"); }
        static std::string grid() { return createIfNotExists("grid/"); }
        static std::string graphs() { return createIfNotExists("graphs/"); }
        static std::string tex() { return createIfNotExists("tex/"); }
        static std::string datasets()
        {
            auto env = platform::DotEnv();
            return env.get("source_data");
        }
        static std::string experiment_file(const std::string& fileName, bool discretize, bool stratified, int seed, int nfold)
        {
            std::string disc = discretize ? "_disc_" : "_ndisc_";
            std::string strat = stratified ? "strat_" : "nstrat_";
            return "datasets_experiment/" + fileName + disc + strat + std::to_string(seed) + "_" + std::to_string(nfold) + ".json";
        }
        static void createPath(const std::string& path)
        {
            // Create directory if it does not exist
            try {
                std::filesystem::create_directory(path);
            }
            catch (std::exception& e) {
                throw std::runtime_error("Could not create directory " + path);
            }
        }
        static std::string bestResultsFile(const std::string& score, const std::string& model)
        {
            return "best_results_" + score + "_" + model + ".json";
        }
        static std::string bestResultsExcel(const std::string& score)
        {
            return "BestResults_" + score + ".xlsx";
        }
        static std::string excelResults() { return "some_results.xlsx"; }
        static std::string grid_input(const std::string& model)
        {
            return grid() + "grid_" + model + "_input.json";
        }
        static std::string grid_output(const std::string& model)
        {
            return grid() + "grid_" + model + "_output.json";
        }
        static std::string tex_output()
        {
            return "results.tex";
        }
        static std::string tex_post_hoc()
        {
            return "post_hoc.tex";
        }
    };
}
#endif