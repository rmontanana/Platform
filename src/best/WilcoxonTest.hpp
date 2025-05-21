#ifndef BEST_WILCOXON_TEST_HPP
#define BEST_WILCOXON_TEST_HPP
// WilcoxonTest.hpp
// Stand‑alone class for paired Wilcoxon signed‑rank post‑hoc analysis
// ------------------------------------------------------------------
//  * Constructor takes the *already‑loaded* nlohmann::json object plus the
//    vectors of model and dataset names.
//  * Internally selects a control model (highest average AUC) and builds all
//    statistics (ranks, W/T/L counts, Wilcoxon p‑values).
//  * Public API:
//        int                  getControlIdx()      const;
//        PostHocResult        getPostHocResult()   const;
//
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <nlohmann/json.hpp>
#include "Statistics.h"

namespace platform {
    class WilcoxonTest {
    public:
        WilcoxonTest(const std::vector<std::string>& models,
            const std::vector<std::string>& datasets,
            const json& data,
            double                          alpha = 0.05)
            : models_(models), datasets_(datasets), data_(data), alpha_(alpha)
        {
            buildAUCTable();              // extracts all AUCs into a dense matrix
            computeAverageAUCs();         // per‑model mean (→ control selection)
            computeAverageRanks();        // Friedman‑style ranks per model
            selectControlModel();         // sets control_idx_
            buildPostHocResult();         // fills postHocResult_
        }

        //---------------------------------------------------- public API ----
        int getControlIdx() const noexcept { return control_idx_; }

        const PostHocResult& getPostHocResult() const noexcept { return postHocResult_; }

    private:
        //-------------------------------------------------- helper structs ----
        // When a value is missing we keep NaN so that ordinary arithmetic still
        // works (NaN simply propagates and we can test with std::isnan).
        using Matrix = std::vector<std::vector<double>>; // [model][dataset]

        //------------------------------------------------- implementation ----
        void buildAUCTable()
        {
            const std::size_t M = models_.size();
            const std::size_t D = datasets_.size();
            auc_.assign(M, std::vector<double>(D, std::numeric_limits<double>::quiet_NaN()));

            for (std::size_t i = 0; i < M; ++i) {
                const auto& model = models_[i];
                for (std::size_t j = 0; j < D; ++j) {
                    const auto& ds = datasets_[j];
                    try {
                        auc_[i][j] = data_.at(model).at(ds).at(0).get<double>();
                    }
                    catch (...) {
                        // leave as NaN when value missing
                    }
                }
            }
        }

        void computeAverageAUCs()
        {
            const std::size_t M = models_.size();
            avg_auc_.resize(M, std::numeric_limits<double>::quiet_NaN());

            for (std::size_t i = 0; i < M; ++i) {
                double sum = 0.0;
                std::size_t cnt = 0;
                for (double v : auc_[i]) {
                    if (!std::isnan(v)) { sum += v; ++cnt; }
                }
                avg_auc_[i] = cnt ? sum / cnt : std::numeric_limits<double>::quiet_NaN();
            }
        }

        // Average rank across datasets (1 = best).
        void computeAverageRanks()
        {
            const std::size_t M = models_.size();
            const std::size_t D = datasets_.size();
            rank_sum_.assign(M, 0.0);
            rank_cnt_.assign(M, 0);

            const double EPS = 1e-10;

            for (std::size_t j = 0; j < D; ++j) {
                // Collect present values for this dataset
                std::vector<std::pair<double, std::size_t>> vals; // (auc, model_idx)
                vals.reserve(M);
                for (std::size_t i = 0; i < M; ++i) {
                    if (!std::isnan(auc_[i][j]))
                        vals.emplace_back(auc_[i][j], i);
                }
                if (vals.empty()) continue; // no info for this dataset

                // Sort descending (higher AUC better)
                std::sort(vals.begin(), vals.end(), [](auto a, auto b) {
                    return a.first > b.first;
                    });

                // Assign ranks with average for ties
                std::size_t k = 0;
                while (k < vals.size()) {
                    std::size_t l = k + 1;
                    while (l < vals.size() && std::fabs(vals[l].first - vals[k].first) < EPS) ++l;
                    const double avg_rank = (k + 1 + l) * 0.5; // average of ranks (1‑based)
                    for (std::size_t m = k; m < l; ++m) {
                        const auto idx = vals[m].second;
                        rank_sum_[idx] += avg_rank;
                        ++rank_cnt_[idx];
                    }
                    k = l;
                }
            }

            // Final average
            avg_rank_.resize(M, std::numeric_limits<double>::quiet_NaN());
            for (std::size_t i = 0; i < M; ++i) {
                avg_rank_[i] = rank_cnt_[i] ? rank_sum_[i] / rank_cnt_[i]
                    : std::numeric_limits<double>::quiet_NaN();
            }
        }

        void selectControlModel()
        {
            // pick model with highest average AUC (ties → first)
            control_idx_ = 0;
            for (std::size_t i = 1; i < avg_auc_.size(); ++i) {
                if (avg_auc_[i] > avg_auc_[control_idx_]) control_idx_ = static_cast<int>(i);
            }
        }

        void buildPostHocResult()
        {
            const std::size_t M = models_.size();
            const std::size_t D = datasets_.size();
            const std::string& control_name = models_[control_idx_];

            postHocResult_.model = control_name;

            const double practical_threshold = 0.0005; // same heuristic as original code

            for (std::size_t i = 0; i < M; ++i) {
                if (static_cast<int>(i) == control_idx_) continue;

                PostHocLine line;
                line.model = models_[i];
                line.rank = avg_rank_[i];

                WTL wtl;
                std::vector<double> differences;
                differences.reserve(D);

                for (std::size_t j = 0; j < D; ++j) {
                    double auc_control = auc_[control_idx_][j];
                    double auc_other = auc_[i][j];
                    if (std::isnan(auc_control) || std::isnan(auc_other)) continue;

                    double diff = auc_control - auc_other; // control − comparison
                    if (std::fabs(diff) <= practical_threshold) {
                        ++wtl.tie;
                    } else if (diff < 0) {
                        ++wtl.win;   // comparison wins
                    } else {
                        ++wtl.loss; // control wins
                    }
                    differences.push_back(diff);
                }

                line.wtl = wtl;
                line.pvalue = differences.empty() ? 1.0L : static_cast<long double>(wilcoxonSignedRankTest(differences));
                line.reject = (line.pvalue < alpha_);

                postHocResult_.postHocLines.push_back(std::move(line));
            }
        }

        // ------------------------------------------------ Wilcoxon (private) --
        static double wilcoxonSignedRankTest(const std::vector<double>& diffs)
        {
            if (diffs.empty()) return 1.0;

            // Build |diff| + sign vector (exclude zeros)
            struct Node { double absval; int sign; };
            std::vector<Node> v;
            v.reserve(diffs.size());
            for (double d : diffs) {
                if (d != 0.0) v.push_back({ std::fabs(d), d > 0 ? 1 : -1 });
            }
            if (v.empty()) return 1.0;

            // Sort by absolute value
            std::sort(v.begin(), v.end(), [](const Node& a, const Node& b) { return a.absval < b.absval; });

            const double EPS = 1e-10;
            const std::size_t n = v.size();
            std::vector<double> ranks(n, 0.0);

            std::size_t i = 0;
            while (i < n) {
                std::size_t j = i + 1;
                while (j < n && std::fabs(v[j].absval - v[i].absval) < EPS) ++j;
                double avg_rank = (i + 1 + j) * 0.5; // 1‑based ranks
                for (std::size_t k = i; k < j; ++k) ranks[k] = avg_rank;
                i = j;
            }

            double w_plus = 0.0, w_minus = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                if (v[k].sign > 0) w_plus += ranks[k];
                else                w_minus += ranks[k];
            }
            double w = std::min(w_plus, w_minus);
            double mean_w = n * (n + 1) / 4.0;
            double sd_w = std::sqrt(n * (n + 1) * (2 * n + 1) / 24.0);
            if (sd_w == 0.0) return 1.0; // degenerate (all diffs identical)

            double z = (w - mean_w) / sd_w;
            double p_two = std::erfc(std::fabs(z) / std::sqrt(2.0)); // 2‑sided tail
            return p_two;
        }

        //-------------------------------------------------------- data ----
        std::vector<std::string> models_;
        std::vector<std::string> datasets_;
        json                       data_;
        double                     alpha_;

        Matrix                     auc_;         // [model][dataset]
        std::vector<double>        avg_auc_;     // mean AUC per model
        std::vector<double>        avg_rank_;    // mean rank per model
        std::vector<double>        rank_sum_;    // helper for ranks
        std::vector<int>           rank_cnt_;    // datasets counted per model

        int                        control_idx_ = -1;
        PostHocResult              postHocResult_;
    };

} // namespace stats
#endif // BEST_WILCOXON_TEST_HPP