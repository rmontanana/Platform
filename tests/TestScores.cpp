#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <string>
#include "TestUtils.h"
#include "results/Result.h"
#include "common/DotEnv.h"
#include "common/Datasets.h"
#include "common/Paths.h"
#include "main/Scores.h"
#include "config.h"

auto epsilon = 1e-4;

void make_test_bin(int TP, int TN, int FP, int FN, std::vector<int>& y_test, std::vector<int>& y_pred)
{
    // TP
    for (int i = 0; i < TP; i++) {
        y_test.push_back(1);
        y_pred.push_back(1);
    }
    // TN
    for (int i = 0; i < TN; i++) {
        y_test.push_back(0);
        y_pred.push_back(0);
    }
    // FP
    for (int i = 0; i < FP; i++) {
        y_test.push_back(0);
        y_pred.push_back(1);
    }
    // FN
    for (int i = 0; i < FN; i++) {
        y_test.push_back(1);
        y_pred.push_back(0);
    }
}

TEST_CASE("TestScores binary", "[Scores]")
{
    std::vector<int> y_test;
    std::vector<int> y_pred;
    make_test_bin(197, 210, 52, 41, y_test, y_pred);
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    auto y_pred_tensor = torch::tensor(y_pred, torch::kInt32);
    platform::Scores scores(y_test_tensor, y_pred_tensor, 2);
    REQUIRE(scores.accuracy() == Catch::Approx(0.814).epsilon(epsilon));
    REQUIRE(scores.f1_score(0) == Catch::Approx(0.818713));
    REQUIRE(scores.f1_score(1) == Catch::Approx(0.809035));
    REQUIRE(scores.precision(0) == Catch::Approx(0.836653));
    REQUIRE(scores.precision(1) == Catch::Approx(0.791165));
    REQUIRE(scores.recall(0) == Catch::Approx(0.801527));
    REQUIRE(scores.recall(1) == Catch::Approx(0.827731));
    REQUIRE(scores.f1_weighted() == Catch::Approx(0.814106));
    REQUIRE(scores.f1_macro() == Catch::Approx(0.813874));
    auto confusion_matrix = scores.get_confusion_matrix();
    REQUIRE(confusion_matrix[0][0].item<int>() == 210);
    REQUIRE(confusion_matrix[0][1].item<int>() == 52);
    REQUIRE(confusion_matrix[1][0].item<int>() == 41);
    REQUIRE(confusion_matrix[1][1].item<int>() == 197);
}
TEST_CASE("TestScores multiclass", "[Scores]")
{
    std::vector<int> y_test = { 0, 2, 2, 2, 2, 0, 1, 2, 0, 2 };
    std::vector<int> y_pred = { 0, 1, 2, 2, 1, 1, 1, 0, 0, 2 };
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    auto y_pred_tensor = torch::tensor(y_pred, torch::kInt32);
    platform::Scores scores(y_test_tensor, y_pred_tensor, 3);
    REQUIRE(scores.accuracy() == Catch::Approx(0.6).epsilon(epsilon));
    REQUIRE(scores.f1_score(0) == Catch::Approx(0.666667));
    REQUIRE(scores.f1_score(1) == Catch::Approx(0.4));
    REQUIRE(scores.f1_score(2) == Catch::Approx(0.666667));
    REQUIRE(scores.precision(0) == Catch::Approx(0.666667));
    REQUIRE(scores.precision(1) == Catch::Approx(0.25));
    REQUIRE(scores.precision(2) == Catch::Approx(1.0));
    REQUIRE(scores.recall(0) == Catch::Approx(0.666667));
    REQUIRE(scores.recall(1) == Catch::Approx(1.0));
    REQUIRE(scores.recall(2) == Catch::Approx(0.5));
    REQUIRE(scores.f1_weighted() == Catch::Approx(0.64));
    REQUIRE(scores.f1_macro() == Catch::Approx(0.577778));
}
TEST_CASE("Test Confusion Matrix Values", "[Scores]")
{
    std::vector<int> y_test = { 0, 2, 2, 2, 2, 0, 1, 2, 0, 2 };
    std::vector<int> y_pred = { 0, 1, 2, 2, 1, 1, 1, 0, 0, 2 };
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    auto y_pred_tensor = torch::tensor(y_pred, torch::kInt32);
    platform::Scores scores(y_test_tensor, y_pred_tensor, 3);
    auto confusion_matrix = scores.get_confusion_matrix();
    REQUIRE(confusion_matrix[0][0].item<int>() == 2);
    REQUIRE(confusion_matrix[0][1].item<int>() == 1);
    REQUIRE(confusion_matrix[0][2].item<int>() == 0);
    REQUIRE(confusion_matrix[1][0].item<int>() == 0);
    REQUIRE(confusion_matrix[1][1].item<int>() == 1);
    REQUIRE(confusion_matrix[1][2].item<int>() == 0);
    REQUIRE(confusion_matrix[2][0].item<int>() == 1);
    REQUIRE(confusion_matrix[2][1].item<int>() == 2);
    REQUIRE(confusion_matrix[2][2].item<int>() == 3);
}
TEST_CASE("Confusion Matrix JSON", "[Scores]")
{
    std::vector<int> y_test = { 0, 2, 2, 2, 2, 0, 1, 2, 0, 2 };
    std::vector<int> y_pred = { 0, 1, 2, 2, 1, 1, 1, 0, 0, 2 };
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    auto y_pred_tensor = torch::tensor(y_pred, torch::kInt32);
    std::vector<std::string> labels = { "Aeroplane", "Boat", "Car" };
    platform::Scores scores(y_test_tensor, y_pred_tensor, 3, labels);
    auto res_json_int = scores.get_confusion_matrix_json();
    REQUIRE(res_json_int[0][0] == 2);
    REQUIRE(res_json_int[0][1] == 1);
    REQUIRE(res_json_int[0][2] == 0);
    REQUIRE(res_json_int[1][0] == 0);
    REQUIRE(res_json_int[1][1] == 1);
    REQUIRE(res_json_int[1][2] == 0);
    REQUIRE(res_json_int[2][0] == 1);
    REQUIRE(res_json_int[2][1] == 2);
    REQUIRE(res_json_int[2][2] == 3);
    auto res_json_str = scores.get_confusion_matrix_json(true);
    REQUIRE(res_json_str["Aeroplane"][0] == 2);
    REQUIRE(res_json_str["Aeroplane"][1] == 1);
    REQUIRE(res_json_str["Aeroplane"][2] == 0);
    REQUIRE(res_json_str["Boat"][0] == 0);
    REQUIRE(res_json_str["Boat"][1] == 1);
    REQUIRE(res_json_str["Boat"][2] == 0);
    REQUIRE(res_json_str["Car"][0] == 1);
    REQUIRE(res_json_str["Car"][1] == 2);
    REQUIRE(res_json_str["Car"][2] == 3);
}
TEST_CASE("Classification Report", "[Scores]")
{
    std::vector<int> y_test = { 0, 2, 2, 2, 2, 0, 1, 2, 0, 2 };
    std::vector<int> y_pred = { 0, 1, 2, 2, 1, 1, 1, 0, 0, 2 };
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    auto y_pred_tensor = torch::tensor(y_pred, torch::kInt32);
    std::vector<std::string> labels = { "Aeroplane", "Boat", "Car" };
    platform::Scores scores(y_test_tensor, y_pred_tensor, 3, labels);
    std::string expected = R"(Classification Report
=====================
             precision recall    f1-score  support
             ========= ========= ========= =========
   Aeroplane 0.6666667 0.6666667 0.6666667         3
        Boat 0.2500000 1.0000000 0.4000000         1
         Car 1.0000000 0.5000000 0.6666667         6

    accuracy                     0.6000000        10
   macro avg 0.6388889 0.7222223 0.5777778        10
weighted avg 0.8250000 0.6000000 0.6400000        10
)";
    REQUIRE(scores.classification_report() == expected);
}
TEST_CASE("JSON constructor", "[Scores]")
{
    std::vector<int> y_test = { 0, 2, 2, 2, 2, 0, 1, 2, 0, 2 };
    std::vector<int> y_pred = { 0, 1, 2, 2, 1, 1, 1, 0, 0, 2 };
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    auto y_pred_tensor = torch::tensor(y_pred, torch::kInt32);
    std::vector<std::string> labels = { "Aeroplane", "Boat", "Car" };
    platform::Scores scores(y_test_tensor, y_pred_tensor, 3, labels);
    auto res_json_int = scores.get_confusion_matrix_json();
    platform::Scores scores2(res_json_int);
    REQUIRE(scores.accuracy() == scores2.accuracy());
    for (int i = 0; i < 2; ++i) {
        REQUIRE(scores.f1_score(i) == scores2.f1_score(i));
        REQUIRE(scores.precision(i) == scores2.precision(i));
        REQUIRE(scores.recall(i) == scores2.recall(i));
    }
    REQUIRE(scores.f1_weighted() == scores2.f1_weighted());
    REQUIRE(scores.f1_macro() == scores2.f1_macro());
    auto res_json_key = scores.get_confusion_matrix_json(true);
    platform::Scores scores3(res_json_key);
    REQUIRE(scores.accuracy() == scores3.accuracy());
    for (int i = 0; i < 2; ++i) {
        REQUIRE(scores.f1_score(i) == scores3.f1_score(i));
        REQUIRE(scores.precision(i) == scores3.precision(i));
        REQUIRE(scores.recall(i) == scores3.recall(i));
    }
    REQUIRE(scores.f1_weighted() == scores3.f1_weighted());
    REQUIRE(scores.f1_macro() == scores3.f1_macro());
}