#ifndef GRIDCONFIG_H
#define GRIDCONFIG_H
#include <string>
#include <vector>
#include <map>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include "common/Datasets.h"
#include "common/Timer.hpp"
#include "main/HyperParameters.h"
#include "GridData.h"
#include "GridConfig.h"
#include "bayesnet/network/Network.h"


namespace platform {
    using json = nlohmann::ordered_json;
    struct ConfigGrid {
        std::string model;
        std::string score;
        std::string continue_from;
        std::string platform;
        std::string smooth_strategy;
        // Algoritmo de discretización pedido en la línea de órdenes
        // (--discretize-algo). Si queda vacío se usa el valor de discretize_algo
        // del .env, que es el comportamiento de siempre.
        std::string discretize_algo;
        bool quiet = false;
        bool only = false; // used with continue_from to only compute that dataset
        bool discretize = false;
        bool stratified = false;
        // b_grid experiment, as in b_main (Platform#9)
        bool no_train_score = false;
        bool graph = false;
        bool generate_fold_files = false;
        int nested = 0;
        int n_folds = 0;
        json excluded;
        std::vector<int> seeds;
    };
    struct ConfigMPI {
        int rank;
        int n_procs;
        int manager;
    };
    typedef struct {
        uint idx_dataset;
        uint idx_combination;
        int n_fold;
        double score; // Experiment: Score test
        double time; // Experiment: Time test
        double time_train;
        double nodes; // Experiment specific
        double leaves; // Experiment specific
        double depth; // Experiment specific
        int process;
        int task;
        // Experiment only (Platform#8, Platform#9). GridSearch leaves them at 0, and then
        // no message follows the result.
        int seed;
        double score_train;
        int has_train;  // 1 when score_train was computed
        int n_classes;  // > 0: a TAG_CONFUSION message (and TAG_CONFUSION_TRAIN if has_train) of n_classes^2 ints follows
        int notes_len;  // > 0: a TAG_NOTES message of notes_len chars follows (notes separated by '\n')
        int graph_len;  // > 0: a TAG_GRAPH message of graph_len chars follows
    } Task_Result;
    // Data of a result too big or too variable for Task_Result. It travels in the
    // messages that follow TAG_RESULT, from the same worker, in this order.
    struct Task_Payload {
        std::vector<int> cm_test;
        std::vector<int> cm_train;
        std::string notes;
        std::string graph;
    };
    const int TAG_QUERY = 1;
    const int TAG_RESULT = 2;
    const int TAG_TASK = 3;
    const int TAG_END = 4;
    const int TAG_CONFUSION = 5;
    const int TAG_CONFUSION_TRAIN = 6;
    const int TAG_NOTES = 7;
    const int TAG_GRAPH = 8;
} /* namespace platform */
#endif