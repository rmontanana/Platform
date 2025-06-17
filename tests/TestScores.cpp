#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <string>
#include "TestUtils.h"
#include "results/Result.h"
#include "common/DotEnv.h"
#include "common/Datasets.h"
#include "common/Paths.h"
#include "common/Colors.h"
#include "main/Scores.h"
#include "config_platform.h"

using json = nlohmann::ordered_json;
auto epsilon = 1e-4;

void make_test_bin(int TP, int TN, int FP, int FN, std::vector<int>& y_test, torch::Tensor& y_pred)
{
    std::vector<std::array<double, 2>> probs;
    // TP: true positive (label 1, predicted 1)
    for (int i = 0; i < TP; i++) {
        y_test.push_back(1);
        probs.push_back({ 0.0, 1.0 }); // P(class 0)=0, P(class 1)=1
    }
    // TN: true negative (label 0, predicted 0)
    for (int i = 0; i < TN; i++) {
        y_test.push_back(0);
        probs.push_back({ 1.0, 0.0 }); // P(class 0)=1, P(class 1)=0
    }
    // FP: false positive (label 0, predicted 1)
    for (int i = 0; i < FP; i++) {
        y_test.push_back(0);
        probs.push_back({ 0.0, 1.0 }); // P(class 0)=0, P(class 1)=1
    }
    // FN: false negative (label 1, predicted 0)
    for (int i = 0; i < FN; i++) {
        y_test.push_back(1);
        probs.push_back({ 1.0, 0.0 }); // P(class 0)=1, P(class 1)=0
    }
    // Convert to torch::Tensor of double, shape [N,2]
    y_pred = torch::from_blob(probs.data(), { (long)probs.size(), 2 }, torch::kFloat64).clone();
}

TEST_CASE("Scores binary", "[Scores]")
{
    std::vector<int> y_test;
    torch::Tensor y_pred;
    make_test_bin(197, 210, 52, 41, y_test, y_pred);
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    platform::Scores scores(y_test_tensor, y_pred, 2);
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
TEST_CASE("Scores multiclass", "[Scores]")
{
    std::vector<int> y_test = { 0, 2, 2, 2, 2, 0, 1, 2, 0, 2 };
    // Refactor y_pred to a tensor of shape [10, 3] with probabilities
    std::vector<std::array<double, 3>> probs = {
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 0.0, 1.0 } // P(class 0)=0, P(class 1)=0, P(class 2)=1
    };
    torch::Tensor y_pred = torch::from_blob(probs.data(), { (long)probs.size(), 3 }, torch::kFloat64).clone();
    // Convert y_test to a tensor
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    platform::Scores scores(y_test_tensor, y_pred, 3);
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
    std::vector<std::array<double, 3>> probs = {
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 0.0, 1.0 } // P(class 0)=0, P(class 1)=0, P(class 2)=1
    };
    torch::Tensor y_pred = torch::from_blob(probs.data(), { (long)probs.size(), 3 }, torch::kFloat64).clone();
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    platform::Scores scores(y_test_tensor, y_pred, 3);
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
    std::vector<std::array<double, 3>> probs = {
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 0.0, 1.0 } // P(class 0)=0, P(class 1)=0, P(class 2)=1
    };
    torch::Tensor y_pred = torch::from_blob(probs.data(), { (long)probs.size(), 3 }, torch::kFloat64).clone();
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    std::vector<std::string> labels = { "Aeroplane", "Boat", "Car" };
    platform::Scores scores(y_test_tensor, y_pred, 3, labels);
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
    std::vector<std::array<double, 3>> probs = {
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 0.0, 1.0 } // P(class 0)=0, P(class 1)=0, P(class 2)=1
    };
    torch::Tensor y_pred = torch::from_blob(probs.data(), { (long)probs.size(), 3 }, torch::kFloat64).clone();
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    std::vector<std::string> labels = { "Aeroplane", "Boat", "Car" };
    platform::Scores scores(y_test_tensor, y_pred, 3, labels);
    auto report = scores.classification_report(Colors::BLUE(), "train");
    auto json_matrix = scores.get_confusion_matrix_json(true);
    platform::Scores scores2(json_matrix);
    REQUIRE(scores.classification_report() == scores2.classification_report());
}
TEST_CASE("JSON constructor", "[Scores]")
{
    std::vector<int> y_test = { 0, 2, 2, 2, 2, 0, 1, 2, 0, 2 };
    std::vector<std::array<double, 3>> probs = {
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 0.0, 1.0 } // P(class 0)=0, P(class 1)=0, P(class 2)=1
    };
    torch::Tensor y_pred = torch::from_blob(probs.data(), { (long)probs.size(), 3 }, torch::kFloat64).clone();
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    std::vector<std::string> labels = { "Car", "Boat", "Aeroplane" };
    platform::Scores scores(y_test_tensor, y_pred, 3, labels);
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
TEST_CASE("Aggregate", "[Scores]")
{
    std::vector<int> y_test;
    torch::Tensor y_pred;
    make_test_bin(197, 210, 52, 41, y_test, y_pred);
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    platform::Scores scores(y_test_tensor, y_pred, 2);
    y_test.clear();
    make_test_bin(227, 187, 39, 47, y_test, y_pred);
    auto y_test_tensor2 = torch::tensor(y_test, torch::kInt32);
    platform::Scores scores2(y_test_tensor2, y_pred, 2);
    scores.aggregate(scores2);
    REQUIRE(scores.accuracy() == Catch::Approx(0.821).epsilon(epsilon));
    REQUIRE(scores.f1_score(0) == Catch::Approx(0.8160329));
    REQUIRE(scores.f1_score(1) == Catch::Approx(0.8257059));
    REQUIRE(scores.precision(0) == Catch::Approx(0.8185567));
    REQUIRE(scores.precision(1) == Catch::Approx(0.8233010));
    REQUIRE(scores.recall(0) == Catch::Approx(0.8135246));
    REQUIRE(scores.recall(1) == Catch::Approx(0.8281250));
    REQUIRE(scores.f1_weighted() == Catch::Approx(0.8209856));
    REQUIRE(scores.f1_macro() == Catch::Approx(0.8208694));
    y_test.clear();
    make_test_bin(197 + 227, 210 + 187, 52 + 39, 41 + 47, y_test, y_pred);
    y_test_tensor = torch::tensor(y_test, torch::kInt32);
    platform::Scores scores3(y_test_tensor, y_pred, 2);
    for (int i = 0; i < 2; ++i) {
        REQUIRE(scores3.f1_score(i) == scores.f1_score(i));
        REQUIRE(scores3.precision(i) == scores.precision(i));
        REQUIRE(scores3.recall(i) == scores.recall(i));
    }
    REQUIRE(scores3.f1_weighted() == scores.f1_weighted());
    REQUIRE(scores3.f1_macro() == scores.f1_macro());
    REQUIRE(scores3.accuracy() == scores.accuracy());
}
TEST_CASE("Order of keys", "[Scores]")
{
    std::vector<int> y_test = { 0, 2, 2, 2, 2, 0, 1, 2, 0, 2 };
    std::vector<std::array<double, 3>> probs = {
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 0.0, 1.0 }, // P(class 0)=0, P(class 1)=0, P(class 2)=1
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 0.0, 1.0, 0.0 }, // P(class 0)=0, P(class 1)=1, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 1.0, 0.0, 0.0 }, // P(class 0)=1, P(class 1)=0, P(class 2)=0
        { 0.0, 0.0, 1.0 } // P(class 0)=0, P(class 1)=0, P(class 2)=1
    };
    torch::Tensor y_pred = torch::from_blob(probs.data(), { (long)probs.size(), 3 }, torch::kFloat64).clone();
    auto y_test_tensor = torch::tensor(y_test, torch::kInt32);
    std::vector<std::string> labels = { "Car", "Boat", "Aeroplane" };
    platform::Scores scores(y_test_tensor, y_pred, 3, labels);
    auto res_json_int = scores.get_confusion_matrix_json(true);
    // Make a temp file and store the json
    std::string filename = "temp.json";
    std::ofstream file(filename);
    file << res_json_int;
    file.close();
    // Read the json from the file
    std::ifstream file2(filename);
    json res_json_int2;
    file2 >> res_json_int2;
    file2.close();
    // Remove the temp file
    std::remove(filename.c_str());
    platform::Scores scores2(res_json_int2);
    REQUIRE(scores.accuracy() == scores2.accuracy());
    for (int i = 0; i < 2; ++i) {
        REQUIRE(scores.f1_score(i) == scores2.f1_score(i));
        REQUIRE(scores.precision(i) == scores2.precision(i));
        REQUIRE(scores.recall(i) == scores2.recall(i));
    }
}