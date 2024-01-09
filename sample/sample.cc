#include <iostream>
#include <torch/torch.h>
#include <string>
#include <map>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "ArffFiles.h"
#include "BayesMetrics.h"
#include "CPPFImdlp.h"
#include "folding.hpp"
#include "Models.h"
#include "modelRegister.h"
#include <fstream>
#include "config.h"

const std::string PATH = { data_path.begin(), data_path.end() };

pair<std::vector<mdlp::labels_t>, map<std::string, int>> discretize(std::vector<mdlp::samples_t>& X, mdlp::labels_t& y, std::vector<std::string> features)
{
    std::vector<mdlp::labels_t>Xd;
    map<std::string, int> maxes;

    auto fimdlp = mdlp::CPPFImdlp();
    for (int i = 0; i < X.size(); i++) {
        fimdlp.fit(X[i], y);
        mdlp::labels_t& xd = fimdlp.transform(X[i]);
        maxes[features[i]] = *max_element(xd.begin(), xd.end()) + 1;
        Xd.push_back(xd);
    }
    return { Xd, maxes };
}

bool file_exists(const std::string& name)
{
    if (FILE* file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}
pair<std::vector<std::vector<int>>, std::vector<int>> extract_indices(std::vector<int> indices, std::vector<std::vector<int>> X, std::vector<int> y)
{
    std::vector<std::vector<int>> Xr; // nxm
    std::vector<int> yr;
    for (int col = 0; col < X.size(); ++col) {
        Xr.push_back(std::vector<int>());
    }
    for (auto index : indices) {
        for (int col = 0; col < X.size(); ++col) {
            Xr[col].push_back(X[col][index]);
        }
        yr.push_back(y[index]);
    }
    return { Xr, yr };
}

int main(int argc, char** argv)
{
    map<std::string, bool> datasets = {
            {"diabetes",           true},
            {"ecoli",              true},
            {"glass",              true},
            {"iris",               true},
            {"kdd_JapaneseVowels", false},
            {"letter",             true},
            {"liver-disorders",    true},
            {"mfeat-factors",      true},
    };
    auto valid_datasets = std::vector<std::string>();
    transform(datasets.begin(), datasets.end(), back_inserter(valid_datasets),
        [](const pair<std::string, bool>& pair) { return pair.first; });
    argparse::ArgumentParser program("PlatformSample");
    program.add_argument("-d", "--dataset")
        .help("Dataset file name")
        .action([valid_datasets](const std::string& value) {
        if (find(valid_datasets.begin(), valid_datasets.end(), value) != valid_datasets.end()) {
            return value;
        }
        throw runtime_error("file must be one of {diabetes, ecoli, glass, iris, kdd_JapaneseVowels, letter, liver-disorders, mfeat-factors}");
            }
    );
    program.add_argument("-p", "--path")
        .help(" folder where the data files are located, default")
        .default_value(std::string{ PATH }
    );
    program.add_argument("-m", "--model")
        .help("Model to use " + platform::Models::instance()->tostring())
        .action([](const std::string& value) {
        static const std::vector<std::string> choices = platform::Models::instance()->getNames();
        if (find(choices.begin(), choices.end(), value) != choices.end()) {
            return value;
        }
        throw runtime_error("Model must be one of " + platform::Models::instance()->tostring());
            }
    );
    program.add_argument("--discretize").help("Discretize input dataset").default_value(false).implicit_value(true);
    program.add_argument("--dumpcpt").help("Dump CPT Tables").default_value(false).implicit_value(true);
    program.add_argument("--stratified").help("If Stratified KFold is to be done").default_value(false).implicit_value(true);
    program.add_argument("--tensors").help("Use tensors to store samples").default_value(false).implicit_value(true);
    program.add_argument("-f", "--folds").help("Number of folds").default_value(5).scan<'i', int>().action([](const std::string& value) {
        try {
            auto k = stoi(value);
            if (k < 2) {
                throw runtime_error("Number of folds must be greater than 1");
            }
            return k;
        }
        catch (const runtime_error& err) {
            throw runtime_error(err.what());
        }
        catch (...) {
            throw runtime_error("Number of folds must be an integer");
        }});
    program.add_argument("-s", "--seed").help("Random seed").default_value(-1).scan<'i', int>();
    bool class_last, stratified, tensors, dump_cpt;
    std::string model_name, file_name, path, complete_file_name;
    int nFolds, seed;
    try {
        program.parse_args(argc, argv);
        file_name = program.get<std::string>("dataset");
        path = program.get<std::string>("path");
        model_name = program.get<std::string>("model");
        complete_file_name = path + file_name + ".arff";
        stratified = program.get<bool>("stratified");
        tensors = program.get<bool>("tensors");
        nFolds = program.get<int>("folds");
        seed = program.get<int>("seed");
        dump_cpt = program.get<bool>("dumpcpt");
        class_last = datasets[file_name];
        if (!file_exists(complete_file_name)) {
            throw runtime_error("Data File " + path + file_name + ".arff" + " does not exist");
        }
    }
    catch (const exception& err) {
        cerr << err.what() << std::endl;
        cerr << program;
        exit(1);
    }

    /*
    * Begin Processing
    */
    auto handler = ArffFiles();
    handler.load(complete_file_name, class_last);
    // Get Dataset X, y
    std::vector<mdlp::samples_t>& X = handler.getX();
    mdlp::labels_t& y = handler.getY();
    // Get className & Features
    auto className = handler.getClassName();
    std::vector<std::string> features;
    auto attributes = handler.getAttributes();
    transform(attributes.begin(), attributes.end(), back_inserter(features),
        [](const pair<std::string, std::string>& item) { return item.first; });
    // Discretize Dataset
    auto [Xd, maxes] = discretize(X, y, features);
    maxes[className] = *max_element(y.begin(), y.end()) + 1;
    map<std::string, std::vector<int>> states;
    for (auto feature : features) {
        states[feature] = std::vector<int>(maxes[feature]);
    }
    states[className] = std::vector<int>(maxes[className]);
    auto clf = platform::Models::instance()->create(model_name);
    clf->fit(Xd, y, features, className, states);
    if (dump_cpt) {
        std::cout << "--- CPT Tables ---" << std::endl;
        clf->dump_cpt();
    }
    auto lines = clf->show();
    for (auto line : lines) {
        std::cout << line << std::endl;
    }
    std::cout << "--- Topological Order ---" << std::endl;
    auto order = clf->topological_order();
    for (auto name : order) {
        std::cout << name << ", ";
    }
    std::cout << "end." << std::endl;
    auto score = clf->score(Xd, y);
    std::cout << "Score: " << score << std::endl;
    auto graph = clf->graph();
    auto dot_file = model_name + "_" + file_name;
    ofstream file(dot_file + ".dot");
    file << graph;
    file.close();
    std::cout << "Graph saved in " << model_name << "_" << file_name << ".dot" << std::endl;
    std::cout << "dot -Tpng -o " + dot_file + ".png " + dot_file + ".dot " << std::endl;
    std::string stratified_string = stratified ? " Stratified" : "";
    std::cout << nFolds << " Folds" << stratified_string << " Cross validation" << std::endl;
    std::cout << "==========================================" << std::endl;
    torch::Tensor Xt = torch::zeros({ static_cast<int>(Xd.size()), static_cast<int>(Xd[0].size()) }, torch::kInt32);
    torch::Tensor yt = torch::tensor(y, torch::kInt32);
    for (int i = 0; i < features.size(); ++i) {
        Xt.index_put_({ i, "..." }, torch::tensor(Xd[i], torch::kInt32));
    }
    float total_score = 0, total_score_train = 0, score_train, score_test;
    folding::Fold* fold;
    if (stratified)
        fold = new folding::StratifiedKFold(nFolds, y, seed);
    else
        fold = new folding::KFold(nFolds, y.size(), seed);
    for (auto i = 0; i < nFolds; ++i) {
        auto [train, test] = fold->getFold(i);
        std::cout << "Fold: " << i + 1 << std::endl;
        if (tensors) {
            auto ttrain = torch::tensor(train, torch::kInt64);
            auto ttest = torch::tensor(test, torch::kInt64);
            torch::Tensor Xtraint = torch::index_select(Xt, 1, ttrain);
            torch::Tensor ytraint = yt.index({ ttrain });
            torch::Tensor Xtestt = torch::index_select(Xt, 1, ttest);
            torch::Tensor ytestt = yt.index({ ttest });
            clf->fit(Xtraint, ytraint, features, className, states);
            auto temp = clf->predict(Xtraint);
            score_train = clf->score(Xtraint, ytraint);
            score_test = clf->score(Xtestt, ytestt);
        } else {
            auto [Xtrain, ytrain] = extract_indices(train, Xd, y);
            auto [Xtest, ytest] = extract_indices(test, Xd, y);
            clf->fit(Xtrain, ytrain, features, className, states);
            score_train = clf->score(Xtrain, ytrain);
            score_test = clf->score(Xtest, ytest);
        }
        if (dump_cpt) {
            std::cout << "--- CPT Tables ---" << std::endl;
            clf->dump_cpt();
        }
        total_score_train += score_train;
        total_score += score_test;
        std::cout << "Score Train: " << score_train << std::endl;
        std::cout << "Score Test : " << score_test << std::endl;
        std::cout << "-------------------------------------------------------------------------------" << std::endl;
    }
    std::cout << "**********************************************************************************" << std::endl;
    std::cout << "Average Score Train: " << total_score_train / nFolds << std::endl;
    std::cout << "Average Score Test : " << total_score / nFolds << std::endl;return 0;
}