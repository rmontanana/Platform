#include <sstream>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include "common/Colors.h"
#include "common/Symbols.h"
#include "common/CLocale.h"
#include "BestResultsTex.h"
#include "BestResultsMd.h"
#include "Statistics.h"
#include "WilcoxonTest.hpp"


namespace platform {

    Statistics::Statistics(const std::string& score, const std::vector<std::string>& models, const std::vector<std::string>& datasets, const json& data, double significance, bool output) :
        score(score), models(models), datasets(datasets), data(data), significance(significance), output(output)
    {
        if (score == "accuracy") {
            postHocType = "Holm";
            hlen = 85;
        } else {
            postHocType = "Wilcoxon";
            hlen = 88;
        }
        nModels = models.size();
        nDatasets = datasets.size();
        auto temp = ConfigLocale();
    }
    void Statistics::fit()
    {
        if (nModels < 3 || nDatasets < 3) {
            std::cerr << "nModels: " << nModels << std::endl;
            std::cerr << "nDatasets: " << nDatasets << std::endl;
            throw std::runtime_error("Can't make the Friedman test with less than 3 models and/or less than 3 datasets.");
        }
        ranksModels.clear();
        computeRanks(); // compute greaterAverage and ranks
        // Set the control model as the one with the lowest average rank
        controlIdx = score == "accuracy" ?
            distance(ranks.begin(), min_element(ranks.begin(), ranks.end(), [](const auto& l, const auto& r) { return l.second < r.second; }))
            : greaterAverage; // The model with the greater average score
        computeWTL();
        maxModelName = (*std::max_element(models.begin(), models.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size();
        maxDatasetName = (*std::max_element(datasets.begin(), datasets.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size();
        fitted = true;
    }
    std::map<std::string, float> assignRanks(std::vector<std::pair<std::string, double>>& ranksOrder)
    {
        // sort the ranksOrder std::vector by value
        std::sort(ranksOrder.begin(), ranksOrder.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
            return a.second > b.second;
            });
        //Assign ranks to  values and if they are the same they share the same averaged rank
        std::map<std::string, float> ranks;
        for (int i = 0; i < ranksOrder.size(); i++) {
            ranks[ranksOrder[i].first] = i + 1.0;
        }
        int i = 0;
        while (i < static_cast<int>(ranksOrder.size())) {
            int j = i + 1;
            int sumRanks = ranks[ranksOrder[i].first];
            while (j < static_cast<int>(ranksOrder.size()) && ranksOrder[i].second == ranksOrder[j].second) {
                sumRanks += ranks[ranksOrder[j++].first];
            }
            if (j > i + 1) {
                float averageRank = (float)sumRanks / (j - i);
                for (int k = i; k < j; k++) {
                    ranks[ranksOrder[k].first] = averageRank;
                }
            }
            i = j;
        }
        return ranks;
    }
    void Statistics::computeRanks()
    {
        std::map<std::string, float> ranksLine;
        std::map<std::string, float> averages;
        for (const auto& model : models) {
            averages[model] = 0;
        }
        for (const auto& dataset : datasets) {
            std::vector<std::pair<std::string, double>> ranksOrder;
            for (const auto& model : models) {
                double value = data[model].at(dataset).at(0).get<double>();
                ranksOrder.push_back({ model, value });
                averages[model] += value;
            }
            // Assign the ranks
            ranksLine = assignRanks(ranksOrder);
            // Store the ranks of the dataset
            ranksModels[dataset] = ranksLine;
            if (ranks.size() == 0) {
                ranks = ranksLine;
            } else {
                for (const auto& rank : ranksLine) {
                    ranks[rank.first] += rank.second;
                }
            }
        }
        // Average the ranks
        for (const auto& rank : ranks) {
            ranks[rank.first] /= nDatasets;
        }
        // Average the scores
        for (const auto& average : averages) {
            averages[average.first] /= nDatasets;
        }
        // Get the model with the greater average score
        greaterAverage = distance(averages.begin(), max_element(averages.begin(), averages.end(), [](const auto& l, const auto& r) { return l.second < r.second; }));
    }
    void Statistics::computeWTL()
    {
        const double practical_threshold = 0.0005;
        // Compute the WTL matrix (Win Tie Loss)
        for (int i = 0; i < nModels; ++i) {
            wtl[i] = { 0, 0, 0 };
        }
        json origin = data.begin().value();
        for (auto const& item : origin.items()) {
            auto controlModel = models.at(controlIdx);
            double controlValue = data[controlModel].at(item.key()).at(0).get<double>();
            for (int i = 0; i < nModels; ++i) {
                if (i == controlIdx) {
                    continue;
                }
                double value = data[models[i]].at(item.key()).at(0).get<double>();
                double diff = controlValue - value; // control − comparison
                if (std::fabs(diff) <= practical_threshold) {
                    wtl[i].tie++;
                } else if (diff < 0) {
                    wtl[i].win++;
                } else {
                    wtl[i].loss++;
                }
            }
        }
    }
    int Statistics::getControlIdx()
    {
        if (!fitted) {
            fit();
        }
        return controlIdx;
    }
    void Statistics::postHocTest()
    {
        if (score == "accuracy") {
            postHocHolmTest();
        } else {
            postHocWilcoxonTest();
        }
    }
    void Statistics::postHocWilcoxonTest()
    {
        if (!fitted) {
            fit();
        }
        // Reference: Wilcoxon, F. (1945). “Individual Comparisons by Ranking Methods”. Biometrics Bulletin, 1(6), 80-83.
        auto wilcoxon = WilcoxonTest(models, datasets, data, significance);
        controlIdx = wilcoxon.getControlIdx();
        postHocResults = wilcoxon.getPostHocResults();
        std::cout << std::string(80, '=') << std::endl;
        setResultsOrder();
        Holm_Bonferroni();
        restoreResultsOrder();
    }
    void Statistics::Holm_Bonferroni()
    {
        // The algorithm need the p-values sorted from the lowest to the highest
        // Sort the models by p-value
        std::sort(postHocResults.begin(), postHocResults.end(), [](const PostHocLine& a, const PostHocLine& b) {
            return a.pvalue < b.pvalue;
            });
        // Holm adjustment
        for (int i = 0; i < postHocResults.size(); ++i) {
            auto item = postHocResults.at(i);
            double before = i == 0 ? 0.0 : postHocResults.at(i - 1).pvalue;
            double p_value = std::min((long double)1.0, item.pvalue * (nModels - i));
            p_value = std::max(before, p_value);
            postHocResults[i].pvalue = p_value;
        }
    }
    void Statistics::setResultsOrder()
    {
        int c = 0;
        for (auto& item : postHocResults) {
            item.idx = c++;
        }

    }
    void Statistics::restoreResultsOrder()
    {
        // Restore the order of the results
        std::sort(postHocResults.begin(), postHocResults.end(), [](const PostHocLine& a, const PostHocLine& b) {
            return a.idx < b.idx;
            });
    }
    void Statistics::postHocHolmTest()
    {
        if (!fitted) {
            fit();
        }
        // Reference https://link.springer.com/article/10.1007/s44196-022-00083-8
        // Post-hoc Holm test
        // Calculate the p-value for the models paired with the control model
        std::map<int, double> stats; // p-value of each model paired with the control model
        boost::math::normal dist(0.0, 1.0);
        double diff = sqrt(nModels * (nModels + 1) / (6.0 * nDatasets));
        for (int i = 0; i < nModels; i++) {
            PostHocLine line;
            line.model = models[i];
            line.rank = ranks.at(models[i]);
            line.wtl = wtl.at(i);
            line.reject = false;
            if (i == controlIdx) {
                postHocResults.push_back(line);
                continue;
            }
            double z = std::abs(ranks.at(models[controlIdx]) - ranks.at(models[i])) / diff;
            line.pvalue = (long double)2 * (1 - cdf(dist, z));
            line.reject = (line.pvalue < significance);
            postHocResults.push_back(line);
        }
        std::sort(postHocResults.begin(), postHocResults.end(), [](const PostHocLine& a, const PostHocLine& b) {
            return a.rank < b.rank;
            });
        setResultsOrder();
        Holm_Bonferroni();
        restoreResultsOrder();
    }

    void Statistics::postHocTestReport(bool friedmanResult, bool tex)
    {

        std::stringstream oss;
        auto color = friedmanResult ? Colors::CYAN() : Colors::YELLOW();
        oss << color;
        oss << "  " << std::string(hlen + 25, '*') << std::endl;
        oss << "  Post-hoc " << postHocType << " test: H0: 'There is no significant differences between the control model and the other models.'" << std::endl;
        oss << "  Control model: " << models.at(controlIdx) << std::endl;
        oss << "  " << std::left << std::setw(maxModelName) << std::string("Model") << " p-value      rank      win tie loss Status" << std::endl;
        oss << "  " << std::string(maxModelName, '=') << " ============ ========= === === ==== =============" << std::endl;
        bool first = true;
        for (const auto& item : postHocResults) {
            if (first) {
                oss << "  " << Colors::BLUE() << std::left << std::setw(maxModelName) << item.model << " ";
                oss << std::setw(12) << " " << std::setprecision(7) << std::fixed << " " << item.rank << std::endl;
                first = false;
                continue;
            }
            auto pvalue = item.pvalue;
            auto colorStatus = pvalue > significance ? Colors::GREEN() : Colors::MAGENTA();
            auto status = pvalue > significance ? Symbols::check_mark : Symbols::cross;
            auto textStatus = pvalue > significance ? " accepted H0" : " rejected H0";
            oss << "  " << colorStatus << std::left << std::setw(maxModelName) << item.model << " ";
            oss << std::setprecision(6) << std::scientific << pvalue << std::setprecision(7) << std::fixed << " " << item.rank;
            oss << " " << std::right << std::setw(3) << item.wtl.win << " " << std::setw(3) << item.wtl.tie << " " << std::setw(4) << item.wtl.loss;
            oss << " " << status << textStatus << std::endl;
        }
        oss << color << "  " << std::string(hlen + 25, '*') << std::endl;
        oss << Colors::RESET();
        if (output) {
            std::cout << oss.str();
        }
        if (tex) {
            BestResultsTex bestResultsTex(score);
            BestResultsMd bestResultsMd;
            bestResultsTex.postHoc_test(postHocResults, postHocType, get_date() + " " + get_time());
            bestResultsMd.postHoc_test(postHocResults, postHocType, get_date() + " " + get_time());
        }
    }
    bool Statistics::friedmanTest()
    {
        if (!fitted) {
            fit();
        }
        std::stringstream oss;
        // Friedman test
        // Calculate the Friedman statistic
        oss << Colors::BLUE() << std::endl;
        oss << std::string(hlen, '*') << std::endl;
        oss << Colors::GREEN() << "Friedman test: H0: 'There is no significant differences between all the classifiers.'" << Colors::BLUE() << std::endl;
        double degreesOfFreedom = nModels - 1.0;
        double sumSquared = 0;
        for (const auto& rank : ranks) {
            sumSquared += pow(rank.second, 2);
        }
        // Compute the Friedman statistic as in https://link.springer.com/article/10.1007/s44196-022-00083-8
        double friedmanQ = 12.0 * nDatasets / (nModels * (nModels + 1)) * (sumSquared - (nModels * pow(nModels + 1, 2)) / 4);
        // Calculate the critical value
        boost::math::chi_squared chiSquared(degreesOfFreedom);
        long double p_value = (long double)1.0 - cdf(chiSquared, friedmanQ);
        double criticalValue = quantile(chiSquared, 1 - significance);
        oss << "Friedman statistic: " << friedmanQ << std::endl;
        oss << "Critical χ2 Value for df=" << std::fixed << (int)degreesOfFreedom
            << " and alpha=" << std::setprecision(2) << std::fixed << significance << ": " << std::setprecision(7) << std::scientific << criticalValue << std::endl;
        oss << "p-value: " << std::scientific << p_value << " is " << (p_value < significance ? "less" : "greater") << " than " << std::setprecision(2) << std::fixed << significance << std::endl;
        bool result;
        if (p_value < significance) {
            oss << Colors::GREEN() << "The null hypothesis H0 is rejected." << std::endl;
            result = true;
        } else {
            oss << Colors::YELLOW() << "The null hypothesis H0 is accepted. Computed p-values will not be significant." << std::endl;
            result = false;
        }
        oss << Colors::BLUE() << std::string(hlen, '*') << Colors::RESET() << std::endl;
        if (output) {
            std::cout << oss.str();
        }
        friedmanResult = { friedmanQ, criticalValue, p_value, result };
        return result;
    }
} // namespace platform
