#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <string>
#include "TestUtils.h"
#include "results/Result.h"
#include "common/DotEnv.h"
#include "common/Datasets.h"
#include "common/Paths.h"
#include "config.h"


TEST_CASE("ZeroR comparison in reports", "[Report]")
{
    auto dotEnv = platform::DotEnv(true);
    auto margin = 1e-4;
    std::string dataset_name = "liver-disorders";
    auto dt = platform::Datasets(false, platform::Paths::datasets());
    auto& dataset = dt.getDataset(dataset_name);
    dataset.load();
    std::vector<int> distribution = dataset.getClassesCounts();
    double nSamples = dataset.getNSamples();
    std::vector<int>::iterator maxValue = max_element(distribution.begin(), distribution.end());
    double mark = *maxValue / nSamples * (1 + margin);
    REQUIRE(mark == Catch::Approx(0.57976811f).epsilon(margin));
}