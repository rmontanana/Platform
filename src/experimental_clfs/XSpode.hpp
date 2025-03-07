#ifndef XSPODE_H
#define XSPODE_H

#include <vector>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <limits>
#include <sstream>
#include <iostream>

namespace platform {

    class XSpode {
    public:
        // --------------------------------------
        // The SPODE can be EMPTY (just created),
        // in COUNTS mode (accumulating raw counts),
        // or in PROBS mode (storing conditional probabilities).
        // --------------------------------------
        enum class MatrixState {
            EMPTY,
            COUNTS,
            PROBS
        };

        // --------------------------------------
        // Constructor
        //
        // Supply which feature index is the single super-parent (“spIndex”).
        // --------------------------------------
        XSpode(int spIndex)
            : superParent_{ spIndex },
            nFeatures_{ 0 },
            statesClass_{ 0 },
            matrixState_{ MatrixState::EMPTY },
            alpha_{ 1.0 }
        {
        }

        // --------------------------------------
        // fit
        // --------------------------------------
        //
        // Trains the SPODE given data:
        //   X: X[f][n] is the f-th feature value for instance n
        //   y: y[n] is the class value for instance n
        //   states: a map or array that tells how many distinct states each feature and the class can take
        //
        // For example, states_.back() is the number of class states,
        // and states_[f] is the number of distinct values for feature f.
        //
        // We only store conditional probabilities for:
        //   p(x_sp| c)   (the super-parent feature)
        //   p(x_child| c, x_sp)  for all child ≠ sp
        //
        // The “weights” can be a vector of per-instance weights; if not used, pass them as 1.0.
        // --------------------------------------
        void fit(const std::vector<std::vector<int>>& X,
            const std::vector<int>& y,
            const torch::Tensor& weights)
        {
            int numInstances = static_cast<int>(y.size());
            nFeatures_ = static_cast<int>(X.size());

            // Derive the number of states for each feature and for the class.
            // (This is just one approach; adapt to match your environment.)
            // Here, we assume the user also gave us the total #states per feature in e.g. statesMap.
            // We'll simply reconstruct the integer states_ array. The last entry is statesClass_.
            states_.resize(nFeatures_);
            for (int f = 0; f < nFeatures_; f++) {
                // Suppose you look up in “statesMap” by the feature name, or read directly from X.
                // We'll assume states_[f] = max value in X[f] + 1.
                auto maxIt = std::max_element(X[f].begin(), X[f].end());
                states_[f] = (*maxIt) + 1;
            }
            // For the class: states_.back() = max(y)+1
            statesClass_ = (*std::max_element(y.begin(), y.end())) + 1;

            // Initialize counts
            classCounts_.resize(statesClass_, 0.0);
            // p(x_sp = spVal | c)
            // We'll store these counts in spFeatureCounts_[spVal * statesClass_ + c].
            spFeatureCounts_.resize(states_[superParent_] * statesClass_, 0.0);

            // For each child ≠ sp, we store p(childVal| c, spVal) in a separate block of childCounts_.
            // childCounts_ will be sized as sum_{child≠sp} (states_[child] * statesClass_ * states_[sp]).
            // We also need an offset for each child to index into childCounts_.
            childOffsets_.resize(nFeatures_, -1);
            int totalSize = 0;
            for (int f = 0; f < nFeatures_; f++) {
                if (f == superParent_) continue; // skip sp
                childOffsets_[f] = totalSize;
                // block size for this child's counts: states_[f] * statesClass_ * states_[superParent_]
                totalSize += (states_[f] * statesClass_ * states_[superParent_]);
            }
            childCounts_.resize(totalSize, 0.0);

            // Switch to COUNTS mode
            matrixState_ = MatrixState::COUNTS;

            // Accumulate raw counts
            for (int n = 0; n < numInstances; n++) {
                std::vector<int> instance(nFeatures_ + 1);
                for (int f = 0; f < nFeatures_; f++) {
                    instance[f] = X[f][n];
                }
                instance[nFeatures_] = y[n];
                addSample(instance, weights[n].item<double>());
            }

            // Laplace smoothing scaled to #instances
            alpha_ = 1.0 / static_cast<double>(numInstances);
            initializer_ = initializer_ = std::numeric_limits<double>::max() / (nFeatures_ * nFeatures_);
            // Convert raw counts to probabilities
            computeProbabilities();
        }

        // --------------------------------------
        // addSample (only valid in COUNTS mode)
        // --------------------------------------
        //
        // instance has size nFeatures_ + 1, with the class at the end.
        // We add 1 to the appropriate counters for each (c, superParentVal, childVal).
        //
        void addSample(const std::vector<int>& instance, double weight)
        {
            if (matrixState_ != MatrixState::COUNTS) {
                throw std::logic_error("addSample: Not in COUNTS mode!");
            }
            if (weight <= 0.0) return;

            int c = instance.back();
            // (A) increment classCounts
            classCounts_[c] += weight;

            // (B) increment super-parent counts => p(x_sp | c)
            int spVal = instance[superParent_];
            spFeatureCounts_[spVal * statesClass_ + c] += weight;

            // (C) increment child counts => p(childVal | c, x_sp)
            for (int f = 0; f < nFeatures_; f++) {
                if (f == superParent_) continue;
                int childVal = instance[f];
                int offset = childOffsets_[f];
                // Compute index in childCounts_.
                // Layout: [ offset + (spVal * states_[f] + childVal) * statesClass_ + c ]
                int blockSize = states_[f] * statesClass_;
                int idx = offset + spVal * blockSize + childVal * statesClass_ + c;
                childCounts_[idx] += weight;
            }
        }

        // --------------------------------------
        // computeProbabilities
        // --------------------------------------
        //
        // Once all samples are added in COUNTS mode, call this to:
        //    p(c)
        //    p(x_sp = spVal | c)
        //    p(x_child = v | c, x_sp = s_sp)
        //
        // We store them in the corresponding *Probs_ arrays for inference.
        // --------------------------------------
        void computeProbabilities()
        {
            if (matrixState_ != MatrixState::COUNTS) {
                throw std::logic_error("computeProbabilities: must be in COUNTS mode.");
            }

            double totalCount = std::accumulate(classCounts_.begin(), classCounts_.end(), 0.0);

            // p(c) => classPriors_
            classPriors_.resize(statesClass_, 0.0);
            if (totalCount <= 0.0) {
                // fallback => uniform
                double unif = 1.0 / static_cast<double>(statesClass_);
                for (int c = 0; c < statesClass_; c++) {
                    classPriors_[c] = unif;
                }
            } else {
                for (int c = 0; c < statesClass_; c++) {
                    classPriors_[c] = (classCounts_[c] + alpha_)
                        / (totalCount + alpha_ * statesClass_);
                }
            }

            // p(x_sp | c)
            spFeatureProbs_.resize(spFeatureCounts_.size());
            // denominator for spVal * statesClass_ + c is just classCounts_[c] + alpha_ * (#states of sp)
            int spCard = states_[superParent_];
            for (int spVal = 0; spVal < spCard; spVal++) {
                for (int c = 0; c < statesClass_; c++) {
                    double denom = classCounts_[c] + alpha_ * spCard;
                    double num = spFeatureCounts_[spVal * statesClass_ + c] + alpha_;
                    spFeatureProbs_[spVal * statesClass_ + c] = (denom <= 0.0 ? 0.0 : num / denom);
                }
            }

            // p(x_child | c, x_sp)
            childProbs_.resize(childCounts_.size());
            for (int f = 0; f < nFeatures_; f++) {
                if (f == superParent_) continue;
                int offset = childOffsets_[f];
                int childCard = states_[f];

                // For each spVal, c, childVal in childCounts_:
                for (int spVal = 0; spVal < spCard; spVal++) {
                    for (int childVal = 0; childVal < childCard; childVal++) {
                        for (int c = 0; c < statesClass_; c++) {
                            int idx = offset + spVal * (childCard * statesClass_)
                                + childVal * statesClass_
                                + c;

                            double num = childCounts_[idx] + alpha_;
                            // denominator = spFeatureCounts_[spVal * statesClass_ + c] + alpha_ * (#states of child)
                            double denom = spFeatureCounts_[spVal * statesClass_ + c]
                                + alpha_ * childCard;
                            childProbs_[idx] = (denom <= 0.0 ? 0.0 : num / denom);
                        }
                    }
                }
            }

            matrixState_ = MatrixState::PROBS;
        }

        // --------------------------------------
        // predict_proba
        // --------------------------------------
        //
        // For a single instance x of dimension nFeatures_:
        //  P(c | x) ∝ p(c) × p(x_sp | c) × ∏(child ≠ sp) p(x_child | c, x_sp).
        //
        // Then we normalize the result.
        // --------------------------------------
        std::vector<double> predict_proba(const std::vector<int>& instance) const
        {
            if (matrixState_ != MatrixState::PROBS) {
                throw std::logic_error("predict_proba: the model is not in PROBS mode.");
            }

            std::vector<double> probs(statesClass_, 0.0);

            // Multiply p(c) × p(x_sp | c)
            int spVal = instance[superParent_];
            for (int c = 0; c < statesClass_; c++) {
                double pc = classPriors_[c];
                double pSpC = spFeatureProbs_[spVal * statesClass_ + c];
                probs[c] = pc * pSpC * initializer_;
            }

            // Multiply by each child’s probability p(x_child | c, x_sp)
            for (int feature = 0; feature < nFeatures_; feature++) {
                if (feature == superParent_) continue;  // skip sp
                int sf = instance[feature];
                int offset = childOffsets_[feature];
                int childCard = states_[feature]; // not used directly, but for clarity
                // Index into childProbs_ = offset + spVal*(childCard*statesClass_) + childVal*statesClass_ + c
                int base = offset + spVal * (childCard * statesClass_) + sf * statesClass_;
                for (int c = 0; c < statesClass_; c++) {
                    probs[c] *= childProbs_[base + c];
                }
            }

            // Normalize
            normalize(probs);
            return probs;
        }

        // --------------------------------------
        // predict
        // --------------------------------------
        //
        // Return the class argmax( P(c|x) ).
        // --------------------------------------
        int predict(const std::vector<int>& instance) const
        {
            auto p = predict_proba(instance);
            return static_cast<int>(std::distance(p.begin(),
                std::max_element(p.begin(), p.end())));
        }
        std::vector<int> predict(const std::vector<std::vector<int>>& X) const
        {
            std::vector<int> preds;
            for (const auto& instance : X) {
                preds.push_back(predict(instance));
            }
            return preds;
        }

        // --------------------------------------
        // Utility: normalize
        // --------------------------------------
        void normalize(std::vector<double>& v) const
        {
            double sum = 0.0;
            for (auto val : v) { sum += val; }
            if (sum <= 0.0) {
                return;
            }
            for (auto& val : v) {
                val /= sum;
            }
        }

        // --------------------------------------
        // debug printing, if desired
        // --------------------------------------
        std::string to_string() const
        {
            std::ostringstream oss;
            oss << "---- SPODE Model ----\n"
                << "nFeatures_  = " << nFeatures_ << "\n"
                << "superParent_ = " << superParent_ << "\n"
                << "statesClass_ = " << statesClass_ << "\n"
                << "matrixState_ = "
                << (matrixState_ == MatrixState::EMPTY ? "EMPTY"
                    : (matrixState_ == MatrixState::COUNTS ? "COUNTS" : "PROBS"))
                << "\n";

            oss << "States: [";
            for (int s : states_) oss << s << " ";
            oss << "]\n";

            oss << "classCounts_: [";
            for (double c : classCounts_) oss << c << " ";
            oss << "]\n";

            oss << "classPriors_: [";
            for (double c : classPriors_) oss << c << " ";
            oss << "]\n";

            oss << "spFeatureCounts_: size = " << spFeatureCounts_.size() << "\n[";
            for (double c : spFeatureCounts_) oss << c << " ";
            oss << "]\n";

            oss << "spFeatureProbs_: size = " << spFeatureProbs_.size() << "\n[";
            for (double c : spFeatureProbs_) oss << c << " ";
            oss << "]\n";

            oss << "childCounts_: size = " << childCounts_.size() << "\n[";
            for (double cc : childCounts_) oss << cc << " ";
            oss << "]\n";

            oss << "childProbs_: size = " << childProbs_.size() << "\n[";
            for (double cp : childProbs_) oss << cp << " ";
            oss << "]\n";

            oss << "childOffsets_: [";
            for (int co : childOffsets_) oss << co << " ";
            oss << "]\n";

            oss << "---------------------\n";
            return oss.str();
        }

    private:
        // --------------------------------------
        // MEMBERS
        // --------------------------------------

        int superParent_;                  // which feature is the single super-parent
        int nFeatures_;
        int statesClass_;
        std::vector<int> states_;          // [states_feat0, ..., states_feat(N-1)] (class not included in this array)

        // Class counts
        std::vector<double> classCounts_;  // [c], accumulative
        std::vector<double> classPriors_;  // [c], after normalization

        // For p(x_sp = spVal | c)
        std::vector<double> spFeatureCounts_; // [spVal * statesClass_ + c]
        std::vector<double> spFeatureProbs_;  // same shape, after normalization

        // For p(x_child = childVal | x_sp = spVal, c)
        // childCounts_ is big enough to hold all child features except sp:
        //   For each child f, we store childOffsets_[f] as the start index, then
        //   childVal, spVal, c => the data.
        std::vector<double> childCounts_;
        std::vector<double> childProbs_;
        std::vector<int>    childOffsets_;

        MatrixState matrixState_;
        double alpha_ = 1.0;
        double initializer_; // for numerical stability
    };

} // namespace platform

#endif // XSPODE_H
