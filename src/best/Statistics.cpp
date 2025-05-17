#include <sstream>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include "common/Colors.h"
#include "common/Symbols.h"
#include "common/CLocale.h"
#include "BestResultsTex.h"
#include "BestResultsMd.h"
#include "Statistics.h"
#include "DeLong.h"


namespace platform {

    Statistics::Statistics(const std::vector<std::string>& models, const std::vector<std::string>& datasets, const json& data, double significance, bool output) :
        models(models), datasets(datasets), data(data), significance(significance), output(output)
    {
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
        computeRanks();
        // Set the control model as the one with the lowest average rank
        controlIdx = distance(ranks.begin(), min_element(ranks.begin(), ranks.end(), [](const auto& l, const auto& r) { return l.second < r.second; }));
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
        for (const auto& dataset : datasets) {
            std::vector<std::pair<std::string, double>> ranksOrder;
            for (const auto& model : models) {
                double value = data[model].at(dataset).at(0).get<double>();
                ranksOrder.push_back({ model, value });
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
    }
    void Statistics::computeWTL()
    {
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
                if (value < controlValue) {
                    wtl[i].win++;
                } else if (value == controlValue) {
                    wtl[i].tie++;
                } else {
                    wtl[i].loss++;
                }
            }
        }
    }
    void Statistics::postHocHolmTest()
    {
        if (!fitted) {
            fit();
        }
        std::stringstream oss;
        // Reference https://link.springer.com/article/10.1007/s44196-022-00083-8
        // Post-hoc Holm test
        // Calculate the p-value for the models paired with the control model
        std::map<int, double> stats; // p-value of each model paired with the control model
        boost::math::normal dist(0.0, 1.0);
        double diff = sqrt(nModels * (nModels + 1) / (6.0 * nDatasets));
        for (int i = 0; i < nModels; i++) {
            if (i == controlIdx) {
                stats[i] = 0.0;
                continue;
            }
            double z = std::abs(ranks.at(models[controlIdx]) - ranks.at(models[i])) / diff;
            double p_value = (long double)2 * (1 - cdf(dist, z));
            stats[i] = p_value;
        }
        // Sort the models by p-value
        for (const auto& stat : stats) {
            postHocData.push_back({ stat.first, stat.second });
        }
        std::sort(postHocData.begin(), postHocData.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
            return a.second < b.second;
            });

        // Holm adjustment
        for (int i = 0; i < postHocData.size(); ++i) {
            auto item = postHocData.at(i);
            double before = i == 0 ? 0.0 : postHocData.at(i - 1).second;
            double p_value = std::min((double)1.0, item.second * (nModels - i));
            p_value = std::max(before, p_value);
            postHocData[i] = { item.first, p_value };
        }
        postHocResult.model = models.at(controlIdx);
    }

    void Statistics::postHocTestReport(const std::string& kind, const std::string& metric, bool friedmanResult, bool tex)
    {

        std::stringstream oss;
        postHocResult.model = models.at(controlIdx);
        auto color = friedmanResult ? Colors::CYAN() : Colors::YELLOW();
        oss << color;
        oss << "  *************************************************************************************************************" << std::endl;
        oss << "  Post-hoc " << kind << " test: H0: 'There is no significant differences between the control model and the other models.'" << std::endl;
        oss << "  Control model: " << models.at(controlIdx) << std::endl;
        oss << "  " << std::left << std::setw(maxModelName) << std::string("Model") << " p-value      rank      win tie loss Status" << std::endl;
        oss << "  " << std::string(maxModelName, '=') << " ============ ========= === === ==== =============" << std::endl;
        // sort ranks from lowest to highest
        std::vector<std::pair<std::string, float>> ranksOrder;
        for (const auto& rank : ranks) {
            ranksOrder.push_back({ rank.first, rank.second });
        }
        std::sort(ranksOrder.begin(), ranksOrder.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
            return a.second < b.second;
            });
        // Show the control model info.
        oss << "  " << Colors::BLUE() << std::left << std::setw(maxModelName) << ranksOrder.at(0).first << " ";
        oss << std::setw(12) << " " << std::setprecision(7) << std::fixed << " " << ranksOrder.at(0).second << std::endl;
        for (const auto& item : ranksOrder) {
            auto idx = distance(models.begin(), find(models.begin(), models.end(), item.first));
            double pvalue = 0.0;
            for (const auto& stat : postHocData) {
                if (stat.first == idx) {
                    pvalue = stat.second;
                }
            }
            postHocResult.postHocLines.push_back({ item.first, pvalue, item.second, wtl.at(idx), pvalue < significance });
            if (item.first == models.at(controlIdx)) {
                continue;
            }
            auto colorStatus = pvalue > significance ? Colors::GREEN() : Colors::MAGENTA();
            auto status = pvalue > significance ? Symbols::check_mark : Symbols::cross;
            auto textStatus = pvalue > significance ? " accepted H0" : " rejected H0";
            oss << "  " << colorStatus << std::left << std::setw(maxModelName) << item.first << " ";
            oss << std::setprecision(6) << std::scientific << pvalue << std::setprecision(7) << std::fixed << " " << item.second;
            oss << " " << std::right << std::setw(3) << wtl.at(idx).win << " " << std::setw(3) << wtl.at(idx).tie << " " << std::setw(4) << wtl.at(idx).loss;
            oss << " " << status << textStatus << std::endl;
        }
        oss << color << "  *************************************************************************************************************" << std::endl;
        oss << Colors::RESET();
        if (output) {
            std::cout << oss.str();
        }
        if (tex) {
            BestResultsTex bestResultsTex(metric);
            BestResultsMd bestResultsMd;
            bestResultsTex.postHoc_test(postHocResult, kind, get_date() + " " + get_time());
            bestResultsMd.postHoc_test(postHocResult, kind, get_date() + " " + get_time());
        }
    }
    // void Statistics::postHocDeLongTest(const std::vector<std::vector<int>>& y_trues,
    //     const std::vector<std::vector<std::vector<double>>>& y_probas,
    //     bool tex)
    // {
    //     std::map<int, double> pvalues;
    //     postHocResult.model = models.at(controlIdx);
    //     postHocResult.postHocLines.clear();

    //     for (size_t i = 0; i < models.size(); ++i) {
    //         if ((int)i == controlIdx) continue;
    //         double acc_p = 0.0;
    //         int valid = 0;
    //         for (size_t d = 0; d < y_trues.size(); ++d) {
    //             try {
    //                 auto result = compareModelsWithDeLong(y_probas[controlIdx][d], y_probas[i][d], y_trues[d]);
    //                 acc_p += result.p_value;
    //                 ++valid;
    //             }
    //             catch (...) {}
    //         }
    //         if (valid > 0) {
    //             pvalues[i] = acc_p / valid;
    //         }
    //     }

    //     std::vector<std::pair<int, double>> sorted_pvalues(pvalues.begin(), pvalues.end());
    //     std::sort(sorted_pvalues.begin(), sorted_pvalues.end(), [](const auto& a, const auto& b) {
    //         return a.second < b.second;
    //         });

    //     std::stringstream oss;
    //     oss << "\n*************************************************************************************************************\n";
    //     oss << "  Post-hoc DeLong-Holm test: H0: 'No significant differences in AUC with control model.'\n";
    //     oss << "  Control model: " << models[controlIdx] << "\n";
    //     oss << "  " << std::left << std::setw(maxModelName) << std::string("Model") << " p-value      Adjusted    Result\n";
    //     oss << "  " << std::string(maxModelName, '=') << " ============ ========== =============\n";

    //     double prev = 0.0;
    //     for (size_t i = 0; i < sorted_pvalues.size(); ++i) {
    //         int idx = sorted_pvalues[i].first;
    //         double raw = sorted_pvalues[i].second;
    //         double adj = std::min(1.0, raw * (models.size() - i - 1));
    //         adj = std::max(prev, adj);
    //         prev = adj;
    //         bool reject = adj < significance;

    //         postHocResult.postHocLines.push_back({ models[idx], adj, 0.0f, {}, reject });

    //         auto color = reject ? Colors::MAGENTA() : Colors::GREEN();
    //         auto status = reject ? Symbols::cross : Symbols::check_mark;
    //         auto textStatus = reject ? " rejected H0" : " accepted H0";
    //         oss << "  " << color << std::left << std::setw(maxModelName) << models[idx] << " ";
    //         oss << std::setprecision(6) << std::scientific << raw << " ";
    //         oss << std::setprecision(6) << std::scientific << adj << " " << status << textStatus << "\n";
    //     }
    //     oss << Colors::CYAN() << "  *************************************************************************************************************\n";
    //     oss << Colors::RESET();
    //     if (output) std::cout << oss.str();
    //     if (tex) {
    //         BestResultsTex bestResultsTex;
    //         BestResultsMd bestResultsMd;
    //         bestResultsTex.holm_test(postHocResult, get_date() + " " + get_time());
    //         bestResultsMd.holm_test(postHocResult, get_date() + " " + get_time());
    //     }
    // }
    bool Statistics::friedmanTest()
    {
        if (!fitted) {
            fit();
        }
        std::stringstream oss;
        // Friedman test
        // Calculate the Friedman statistic
        oss << Colors::BLUE() << std::endl;
        oss << "***************************************************************************************************************" << std::endl;
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
        oss << Colors::BLUE() << "***************************************************************************************************************" << Colors::RESET() << std::endl;
        if (output) {
            std::cout << oss.str();
        }
        friedmanResult = { friedmanQ, criticalValue, p_value, result };
        return result;
    }
    FriedmanResult& Statistics::getFriedmanResult()
    {
        return friedmanResult;
    }
    PostHocResult& Statistics::getPostHocResult()
    {
        return postHocResult;
    }
    std::map<std::string, std::map<std::string, float>>& Statistics::getRanks()
    {
        return ranksModels;
    }
} // namespace platform
