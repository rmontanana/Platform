// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************
// Based on the Geoff. I. Webb A1DE java algorithm
// https://weka.sourceforge.io/packageMetaData/AnDE/Latest.html

#ifndef XAODE_H
#define XAODE_H
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include <torch/torch.h>

namespace platform {
    class Xaode {
    public:
        // -------------------------------------------------------
        // The Xaode can be EMPTY (just created), in COUNTS mode (accumulating raw counts)
        // or PROBS mode (storing conditional probabilities).
        enum class MatrixState {
            EMPTY,
            COUNTS,
            PROBS
        };
        std::vector<double> significance_models;
        Xaode() : nFeatures_{ 0 }, statesClass_{ 0 }, matrixState_{ MatrixState::EMPTY } {}
        // -------------------------------------------------------
        // fit
        // -------------------------------------------------------
        //
        // Classifiers interface
        // all parameter decide if the model is initialized with all the parents active or none of them
        //
        void fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bool all_parents)
        {
            int num_instances = X[0].size();
            int n_features_ = X.size();

            significance_models.resize(n_features_, (all_parents ? 1.0 : 0.0));
            std::vector<int> statesv;
            for (int i = 0; i < n_features_; i++) {
                if (all_parents) active_parents.push_back(i);
                statesv.push_back(*max_element(X[i].begin(), X[i].end()) + 1);
            }
            statesv.push_back(*max_element(y.begin(), y.end()) + 1);
            // std::cout << "* States: " << statesv << std::endl;
            // std::cout << "* Weights: " << weights_ << std::endl;
            // std::cout << "* Instances: " << num_instances << std::endl;
            // std::cout << "* Attributes: " << n_features_ +1 << std::endl;
            // std::cout << "* y: " << y << std::endl;
            // std::cout << "* x shape: " << X.size() << "x" << X[0].size() << std::endl;
            // for (int i = 0; i < n_features_; i++) {
            //     std::cout << "* " << features[i] << ": " << instances[i] << std::endl;
            // }
            // std::cout << "Starting to build the model" << std::endl;
            init(statesv);
            std::vector<int> instance(n_features_ + 1);
            for (int n_instance = 0; n_instance < num_instances; n_instance++) {
                for (int feature = 0; feature < n_features_; feature++) {
                    instance[feature] = X[feature][n_instance];
                }
                instance[n_features_] = y[n_instance];
                addSample(instance, weights[n_instance].item<double>());
            }
            computeProbabilities();
        }
        // -------------------------------------------------------
        // init
        // -------------------------------------------------------
        //
        // states.size() = nFeatures + 1,
        //   where states.back() = number of class states.
        //
        // We'll store:
        //  1) p(x_i=si | c) in classFeatureProbs_
        //  2) p(x_j=sj | c, x_i=si) in data_, with i<j => i is "superparent," j is "child."
        //
        // Internally, in COUNTS mode, data_ accumulates raw counts, then
        // computeProbabilities(...) normalizes them into conditionals.
        //
        void init(const std::vector<int>& states)
        {
            //
            // Check Valid input data
            //
            if (matrixState_ != MatrixState::EMPTY) {
                throw std::logic_error("Xaode: already initialized.");
            }
            states_ = states;
            nFeatures_ = static_cast<int>(states_.size()) - 1;
            if (nFeatures_ < 1) {
                throw std::invalid_argument("Xaode: need at least 1 feature plus class states.");
            }
            statesClass_ = states_.back();
            if (statesClass_ <= 0) {
                throw std::invalid_argument("Xaode: class states must be > 0.");
            }
            //
            // Initialize data structures
            //
            active_parents.resize(nFeatures_);
            int totalStates = std::accumulate(states.begin(), states.end(), 0) - statesClass_;

            // For p(x_i=si | c), we store them in a 1D array classFeatureProbs_ after we compute.
            // We'll need the offsets for each feature i in featureClassOffset_.
            featureClassOffset_.resize(nFeatures_);
            // We'll store p(x_child=sj | c, x_sp=si) for each pair (i<j).
            // So data_(i, si, j, sj, c) indexes into a big 1D array with an offset.
            // For p(x_i=si | c), we store them in a 1D array classFeatureProbs_ after we compute.
            // We'll need the offsets for each feature i in featureClassOffset_.
            featureClassOffset_.resize(nFeatures_);
            pairOffset_.resize(totalStates);
            int feature_offset = 0;
            int runningOffset = 0;
            int feature = 0, index = 0;
            for (int i = 0; i < nFeatures_; ++i) {
                featureClassOffset_[i] = feature_offset;
                feature_offset += states_[i];
                for (int j = 0; j < states_[i]; ++j) {
                    pairOffset_[feature++] = index;
                    index += runningOffset;
                }
                runningOffset += states_[i];
            }
            int totalSize = index * statesClass_;
            data_.resize(totalSize);
            dataOpp_.resize(totalSize);

            classFeatureCounts_.resize(feature_offset * statesClass_);
            classFeatureProbs_.resize(feature_offset * statesClass_);

            // classCounts_[c]
            classCounts_.resize(statesClass_, 0.0);

            matrixState_ = MatrixState::COUNTS;
        }

        // Returns current mode: INIT, COUNTS or PROBS
        MatrixState state() const
        {
            return matrixState_;
        }

        // Optional: print a quick summary
        void show() const
        {
            std::cout << "-------- Xaode.show() --------" << std::endl
                << "- nFeatures = " << nFeatures_ << std::endl
                << "- statesClass = " << statesClass_ << std::endl
                << "- matrixState = " << (matrixState_ == MatrixState::COUNTS ? "COUNTS" : "PROBS") << std::endl;
            std::cout << "- states: size: " << states_.size() << std::endl;
            for (int s : states_) std::cout << s << " "; std::cout << std::endl;
            std::cout << "- classCounts: size: " << classCounts_.size() << std::endl;
            for (double cc : classCounts_) std::cout << cc << " "; std::cout << std::endl;
            std::cout << "- classFeatureCounts: size: " << classFeatureCounts_.size() << std::endl;
            for (double cfc : classFeatureCounts_) std::cout << cfc << " "; std::cout << std::endl;
            std::cout << "- classFeatureProbs: size: " << classFeatureProbs_.size() << std::endl;
            for (double cfp : classFeatureProbs_) std::cout << cfp << " "; std::cout << std::endl;
            std::cout << "- featureClassOffset: size: " << featureClassOffset_.size() << std::endl;
            for (int f : featureClassOffset_) std::cout << f << " "; std::cout << std::endl;
            std::cout << "- pairOffset_: size: " << pairOffset_.size() << std::endl;
            for (int p : pairOffset_) std::cout << p << " "; std::cout << std::endl;
            std::cout << "- data: size: " << data_.size() << std::endl;
            for (double d : data_) std::cout << d << " "; std::cout << std::endl;
            std::cout << "--------------------------------" << std::endl;
        }

        // -------------------------------------------------------
        // addSample (only in COUNTS mode)
        // -------------------------------------------------------
        // 
        // instance should have the class at the end.
        // 
        void addSample(const std::vector<int>& instance, double weight)
        {
            //
            // (A) increment classCounts_
            // (B) increment feature–class counts => for p(x_i|c)
            // (C) increment pair (superparent= i, child= j) counts => data_ 
            //

            // if (matrixState_ != MatrixState::COUNTS) {
            //     throw std::logic_error("addSample: not in COUNTS mode.");
            // }
            // if (static_cast<int>(instance.size()) != nFeatures_ + 1) {
            //     throw std::invalid_argument("addSample: instance.size() must be nFeatures_ + 1.");
            // }

            int c = instance.back();
            // if (c < 0 || c >= statesClass_) {
            //     throw std::out_of_range("addSample: class index out of range.");
            // }
            if (weight <= 0.0) {
                return;
            }
            // (A) increment classCounts_
            classCounts_[c] += weight;

            // (B,C)
            // We'll store raw counts now and turn them into p(child| c, superparent) later.
            int idx, fcIndex, si, sj, i_offset;
            for (int i = 0; i < nFeatures_; ++i) {
                si = instance[i];
                // (B) increment feature–class counts => for p(x_i|c)
                fcIndex = (featureClassOffset_[i] + si) * statesClass_ + c;
                classFeatureCounts_[fcIndex] += weight;
                // (C) increment pair (superparent= i, child= j) counts => data_
                i_offset = pairOffset_[featureClassOffset_[i] + si];
                for (int j = 0; j < i; ++j) {
                    sj = instance[j];
                    idx = (i_offset + featureClassOffset_[j] + sj) * statesClass_ + c;
                    data_[idx] += weight;
                }
            }
        }

        // -------------------------------------------------------
        // computeProbabilities
        // -------------------------------------------------------
        //
        // Once all samples are added in COUNTS mode, call this to:
        //  1) compute p(x_i=si | c) => classFeatureProbs_
        //  2) compute p(x_j=sj | c, x_i=si) => data_ (for i<j) dataOpp_ (for i>j)
        //
        void computeProbabilities()
        {
            if (matrixState_ != MatrixState::COUNTS) {
                throw std::logic_error("computeProbabilities: must be in COUNTS mode.");
            }
            double totalCount = std::accumulate(classCounts_.begin(), classCounts_.end(), 0.0);
            // (1) p(x_i=si | c) => classFeatureProbs_
            int idx, sf;
            double denom, countVal, p;
            for (int feature = 0; feature < nFeatures_; ++feature) {
                sf = states_[feature];
                for (int c = 0; c < statesClass_; ++c) {
                    denom = classCounts_[c] * sf;
                    if (denom <= 0.0) {
                        // fallback => uniform
                        for (int sf_value = 0; sf_value < sf; ++sf_value) {
                            idx = (featureClassOffset_[feature] + sf_value) * statesClass_ + c;
                            classFeatureProbs_[idx] = 1.0 / sf;
                        }
                    } else {
                        for (int sf_value = 0; sf_value < sf; ++sf_value) {
                            idx = (featureClassOffset_[feature] + sf_value) * statesClass_ + c;
                            countVal = classFeatureCounts_[idx];
                            p = ((countVal + SMOOTHING / (statesClass_ * states_[feature])) / (totalCount + SMOOTHING));
                            classFeatureProbs_[idx] = p;
                        }
                    }
                }
            }
            // getCountFromTable(int classVal, int pIndex, int childIndex)
            // (2) p(x_j=sj | c, x_i=si) => data_(i,si,j,sj,c)
            // (2) p(x_i=si | c, x_j=sj) => dataOpp_(j,sj,i,si,c)
            double pccCount, pcCount, ccCount;
            double conditionalProb, oppositeCondProb;
            int part1, part2, p1, part2_class, p1_class;
            for (int parent = nFeatures_ - 1; parent >= 0; --parent) {
                // for (int parent = 3; parent >= 3; --parent) {
                for (int sp = 0; sp < states_[parent]; ++sp) {
                    p1 = featureClassOffset_[parent] + sp;
                    part1 = pairOffset_[p1];
                    p1_class = p1 * statesClass_;
                    for (int child = parent - 1; child >= 0; --child) {
                        // for (int child = 2; child >= 2; --child) {
                        for (int sc = 0; sc < states_[child]; ++sc) {
                            part2 = featureClassOffset_[child] + sc;
                            part2_class = part2 * statesClass_;
                            for (int c = 0; c < statesClass_; c++) {
                                //idx = compute_index(parent, sp, child, sc, classval);
                                idx = (part1 + part2) * statesClass_ + c;
                                // Parent, Child, Class Count
                                pccCount = data_[idx];
                                // Parent, Class count
                                pcCount = classFeatureCounts_[p1_class + c];
                                // Child, Class count
                                ccCount = classFeatureCounts_[part2_class + c];
                                conditionalProb = (pccCount + SMOOTHING / states_[parent]) / (ccCount + SMOOTHING);
                                data_[idx] = conditionalProb;
                                oppositeCondProb = (pccCount + SMOOTHING / states_[child]) / (pcCount + SMOOTHING);
                                dataOpp_[idx] = oppositeCondProb;
                            }
                        }
                    }
                }
            }
            matrixState_ = MatrixState::PROBS;
        }

        // -------------------------------------------------------
        // predict_proba_spode
        // -------------------------------------------------------
        //
        // Single-superparent approach:
        // P(c | x) ∝ p(c) * p(x_sp| c) * ∏_{i≠sp} p(x_i | c, x_sp)
        //
        // 'instance' should have size == nFeatures_ (no class).
        // sp in [0..nFeatures_).
        // We multiply p(c) * p(x_sp| c) * p(x_i| c, x_sp).
        // Then normalize the distribution.
        //
        std::vector<double> predict_proba_spode(const std::vector<int>& instance, int parent)
        {
            // accumulates posterior probabilities for each class
            auto probs = std::vector<double>(statesClass_);
            auto spodeProbs = std::vector<double>(statesClass_);
            // Initialize the probabilities with the feature|class probabilities
            int localOffset;
            int sp = instance[parent];
            localOffset = (featureClassOffset_[parent] + sp) * statesClass_;
            for (int c = 0; c < statesClass_; ++c) {
                spodeProbs[c] = classFeatureProbs_[localOffset + c];
            }
            int idx, base, sc, parent_offset;
            sp = instance[parent];
            parent_offset = pairOffset_[featureClassOffset_[parent] + sp];
            for (int child = 0; child < parent; ++child) {
                sc = instance[child];
                base = (parent_offset + featureClassOffset_[child] + sc) * statesClass_;
                for (int c = 0; c < statesClass_; ++c) {
                    /*
                     * The probability P(xc|xp,c) is stored in dataOpp_, and
                     * the probability P(xp|xc,c) is stored in data_
                     */
                     /*
                        int base = pairOffset_[i * nFeatures_ + j];
                        int blockSize = states_[i] * states_[j];
                        return base + c * blockSize + (si * states_[j] + sj);
                     */
                     // index = compute_index(parent, instance[parent], child, instance[child], classVal);
                    idx = base + c;
                    spodeProbs[c] *= data_[idx];
                    spodeProbs[c] *= dataOpp_[idx];
                }
            }
            // Normalize the probabilities
            normalize(probs);
            return probs;
        }
        int predict_spode(const std::vector<int>& instance, int parent)
        {
            auto probs = predict_proba_spode(instance, parent);
            return (int)std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        }
        std::vector<double> predict_proba(const std::vector<int>& instance)
        {
            // accumulates posterior probabilities for each class
            auto probs = std::vector<double>(statesClass_);
            auto spodeProbs = std::vector<std::vector<double>>(nFeatures_, std::vector<double>(statesClass_));
            // Initialize the probabilities with the feature|class probabilities
            int localOffset;
            for (int feature = 0; feature < nFeatures_; ++feature) {
                // if feature is not in the active_parents, skip it
                if (std::find(active_parents.begin(), active_parents.end(), feature) == active_parents.end()) {
                    continue;
                }
                localOffset = (featureClassOffset_[feature] + instance[feature]) * statesClass_;
                for (int c = 0; c < statesClass_; ++c) {
                    spodeProbs[feature][c] = classFeatureProbs_[localOffset + c];
                }
            }
            int idx, base, sp, sc, parent_offset;
            for (int parent = 1; parent < nFeatures_; ++parent) {
                // if parent is not in the active_parents, skip it
                if (std::find(active_parents.begin(), active_parents.end(), parent) == active_parents.end()) {
                    continue;
                }
                sp = instance[parent];
                parent_offset = pairOffset_[featureClassOffset_[parent] + sp];
                for (int child = 0; child < parent; ++child) {
                    sc = instance[child];
                    base = (parent_offset + featureClassOffset_[child] + sc) * statesClass_;
                    for (int c = 0; c < statesClass_; ++c) {
                        /*
                         * The probability P(xc|xp,c) is stored in dataOpp_, and
                         * the probability P(xp|xc,c) is stored in data_
                         */
                         /*
                            int base = pairOffset_[i * nFeatures_ + j];
                            int blockSize = states_[i] * states_[j];
                            return base + c * blockSize + (si * states_[j] + sj);
                         */
                         // index = compute_index(parent, instance[parent], child, instance[child], classVal);
                        idx = base + c;
                        spodeProbs[child][c] *= data_[idx];
                        // spodeProbs[child][c] *= data_.at(index);
                        spodeProbs[parent][c] *= dataOpp_[idx];
                        // spodeProbs[parent][c] *= dataOpp_.at(index);
                    }
                }
            }
            /* add all the probabilities for each class */
            for (int c = 0; c < statesClass_; ++c) {
                for (int i = 0; i < nFeatures_; ++i) {
                    probs[c] += spodeProbs[i][c];
                }
            }
            // Normalize the probabilities
            normalize(probs);
            return probs;
        }
        void normalize(std::vector<double>& probs) const
        {
            double sum = 0;
            for (double d : probs) {
                sum += d;
            }
            if (std::isnan(sum)) {
                throw std::runtime_error("Can't normalize array. Sum is NaN.");
            }
            if (sum == 0) {
                return;
            }
            for (int i = 0; i < (int)probs.size(); i++) {
                probs[i] /= sum;
            }
        }

        int statesClass() const
        {
            return statesClass_;
        }
        int nFeatures() const
        {
            return nFeatures_;
        }
        int getNumberOfStates() const
        {
            return std::accumulate(states_.begin(), states_.end(), 0) * nFeatures_;
        }
        int getNumberOfEdges() const
        {
            return nFeatures_ * (2 * nFeatures_ - 1);
        }
        int getNumberOfNodes() const
        {
            return (nFeatures_ + 1) * nFeatures_;
        }
        void add_active_parent(int active_parent)
        {
            active_parents.push_back(active_parent);
        }
        void remove_last_parent()
        {
            active_parents.pop_back();
        }


    private:
        // -----------
        // MEMBER DATA
        // -----------
        std::vector<int> states_;            // [states_feat0, ..., states_feat(n-1), statesClass_]
        int nFeatures_;
        int statesClass_;

        // data_ means p(child=sj | c, superparent= si) after normalization.
        // But in COUNTS mode, it accumulates raw counts.
        std::vector<int> pairOffset_;
        // data_ stores p(child=sj | c, superparent=si) for each pair (i<j).
        std::vector<double> data_;
        // dataOpp_ stores p(superparent=si | c, child=sj) for each pair (i<j).
        std::vector<double> dataOpp_;

        // classCounts_[c]
        std::vector<double> classCounts_;

        // For p(x_i=si| c), we store counts in classFeatureCounts_ => offset by featureClassOffset_[i]
        std::vector<int> featureClassOffset_;
        std::vector<double> classFeatureCounts_;
        std::vector<double> classFeatureProbs_;  // => p(x_i=si | c) after normalization

        MatrixState matrixState_;

        double SMOOTHING = 1.0;
        std::vector<int> active_parents;
    };
}
#endif // XAODE_H