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
    auto margin = 1e-2;
    std::string dataset = "liver-disorders";
    auto dt = platform::Datasets(false, platform::Paths::datasets());
    dt.loadDataset(dataset);
    std::vector<int> distribution = dt.getClassesCounts(dataset);
    double nSamples = dt.getNSamples(dataset);
    std::vector<int>::iterator maxValue = max_element(distribution.begin(), distribution.end());
    double mark = *maxValue / nSamples * (1 + margin);
    REQUIRE(mark == Catch::Approx(0.585507f).epsilon(1e-5));
}