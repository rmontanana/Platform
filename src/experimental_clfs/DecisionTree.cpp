// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "DecisionTree.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <limits>
#include "TensorUtils.hpp"

namespace bayesnet {

    DecisionTree::DecisionTree(int max_depth, int min_samples_split, int min_samples_leaf)
        : Classifier(Network()), max_depth(max_depth),
        min_samples_split(min_samples_split), min_samples_leaf(min_samples_leaf)
    {
        validHyperparameters = { "max_depth", "min_samples_split", "min_samples_leaf" };
    }

    void DecisionTree::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        // Set hyperparameters from JSON
        auto it = hyperparameters.find("max_depth");
        if (it != hyperparameters.end()) {
            max_depth = it->get<int>();
            hyperparameters.erase("max_depth");  // Remove 'order' if present
        }

        it = hyperparameters.find("min_samples_split");
        if (it != hyperparameters.end()) {
            min_samples_split = it->get<int>();
            hyperparameters.erase("min_samples_split");  // Remove 'min_samples_split' if present   
        }

        it = hyperparameters.find("min_samples_leaf");
        if (it != hyperparameters.end()) {
            min_samples_leaf = it->get<int>();
            hyperparameters.erase("min_samples_leaf");  // Remove 'min_samples_leaf' if present
        }
        Classifier::setHyperparameters(hyperparameters);
        checkValues();
    }
    void DecisionTree::checkValues()
    {
        if (max_depth <= 0) {
            throw std::invalid_argument("max_depth must be positive");
        }
        if (min_samples_leaf <= 0) {
            throw std::invalid_argument("min_samples_leaf must be positive");
        }
        if (min_samples_split <= 0) {
            throw std::invalid_argument("min_samples_split must be positive");
        }
    }
    void DecisionTree::buildModel(const torch::Tensor& weights)
    {
        // Extract features (X) and labels (y) from dataset
        auto X = dataset.index({ torch::indexing::Slice(0, dataset.size(0) - 1), torch::indexing::Slice() }).t();
        auto y = dataset.index({ -1, torch::indexing::Slice() });

        if (X.size(0) != y.size(0)) {
            throw std::runtime_error("X and y must have the same number of samples");
        }

        n_classes = states[className].size();

        // Use provided weights or uniform weights
        torch::Tensor sample_weights;
        if (weights.defined() && weights.numel() > 0) {
            if (weights.size(0) != X.size(0)) {
                throw std::runtime_error("weights must have the same length as number of samples");
            }
            sample_weights = weights;
        } else {
            sample_weights = torch::ones({ X.size(0) }) / X.size(0);
        }

        // Normalize weights
        sample_weights = sample_weights / sample_weights.sum();

        // Build the tree
        root = buildTree(X, y, sample_weights, 0);

        // Mark as fitted
        fitted = true;
    }
    bool DecisionTree::validateTensors(const torch::Tensor& X, const torch::Tensor& y,
        const torch::Tensor& sample_weights) const
    {
        if (X.size(0) != y.size(0) || X.size(0) != sample_weights.size(0)) {
            return false;
        }
        if (X.size(0) == 0) {
            return false;
        }
        return true;
    }

    std::unique_ptr<TreeNode> DecisionTree::buildTree(
        const torch::Tensor& X,
        const torch::Tensor& y,
        const torch::Tensor& sample_weights,
        int current_depth)
    {
        auto node = std::make_unique<TreeNode>();
        int n_samples = y.size(0);

        // Check stopping criteria
        auto unique = at::_unique(y);
        bool should_stop = (current_depth >= max_depth) ||
            (n_samples < min_samples_split) ||
            (std::get<0>(unique).size(0) == 1);  // All samples same class

        if (should_stop || n_samples <= min_samples_leaf) {
            // Create leaf node
            node->is_leaf = true;

            // Calculate class probabilities
            node->class_probabilities = torch::zeros({ n_classes });

            for (int i = 0; i < n_samples; i++) {
                int class_idx = y[i].item<int>();
                node->class_probabilities[class_idx] += sample_weights[i].item<float>();
            }

            // Normalize probabilities
            node->class_probabilities /= node->class_probabilities.sum();

            // Set predicted class as the one with highest probability
            node->predicted_class = torch::argmax(node->class_probabilities).item<int>();

            return node;
        }

        // Find best split
        SplitInfo best_split = findBestSplit(X, y, sample_weights);

        // If no valid split found, create leaf
        if (best_split.feature_index == -1 || best_split.impurity_decrease <= 0) {
            node->is_leaf = true;

            // Calculate class probabilities
            node->class_probabilities = torch::zeros({ n_classes });

            for (int i = 0; i < n_samples; i++) {
                int class_idx = y[i].item<int>();
                node->class_probabilities[class_idx] += sample_weights[i].item<float>();
            }

            node->class_probabilities /= node->class_probabilities.sum();
            node->predicted_class = torch::argmax(node->class_probabilities).item<int>();

            return node;
        }

        // Create internal node
        node->is_leaf = false;
        node->split_feature = best_split.feature_index;
        node->split_value = best_split.split_value;

        // Split data
        auto left_X = X.index({ best_split.left_mask });
        auto left_y = y.index({ best_split.left_mask });
        auto left_weights = sample_weights.index({ best_split.left_mask });

        auto right_X = X.index({ best_split.right_mask });
        auto right_y = y.index({ best_split.right_mask });
        auto right_weights = sample_weights.index({ best_split.right_mask });

        // Recursively build subtrees
        if (left_X.size(0) >= min_samples_leaf) {
            node->left = buildTree(left_X, left_y, left_weights, current_depth + 1);
        } else {
            // Force leaf if not enough samples
            node->left = std::make_unique<TreeNode>();
            node->left->is_leaf = true;
            auto mode = std::get<0>(torch::mode(left_y));
            node->left->predicted_class = mode.item<int>();
            node->left->class_probabilities = torch::zeros({ n_classes });
            node->left->class_probabilities[node->left->predicted_class] = 1.0;
        }

        if (right_X.size(0) >= min_samples_leaf) {
            node->right = buildTree(right_X, right_y, right_weights, current_depth + 1);
        } else {
            // Force leaf if not enough samples
            node->right = std::make_unique<TreeNode>();
            node->right->is_leaf = true;
            auto mode = std::get<0>(torch::mode(right_y));
            node->right->predicted_class = mode.item<int>();
            node->right->class_probabilities = torch::zeros({ n_classes });
            node->right->class_probabilities[node->right->predicted_class] = 1.0;
        }

        return node;
    }

    DecisionTree::SplitInfo DecisionTree::findBestSplit(
        const torch::Tensor& X,
        const torch::Tensor& y,
        const torch::Tensor& sample_weights)
    {

        SplitInfo best_split;
        best_split.feature_index = -1;
        best_split.split_value = -1;
        best_split.impurity_decrease = -std::numeric_limits<double>::infinity();

        int n_features = X.size(1);
        int n_samples = X.size(0);

        // Calculate impurity of current node
        double current_impurity = calculateGiniImpurity(y, sample_weights);
        double total_weight = sample_weights.sum().item<double>();

        // Try each feature
        for (int feat_idx = 0; feat_idx < n_features; feat_idx++) {
            auto feature_values = X.index({ torch::indexing::Slice(), feat_idx });
            auto unique_values = std::get<0>(torch::unique_consecutive(std::get<0>(torch::sort(feature_values))));

            // Try each unique value as split point
            for (int i = 0; i < unique_values.size(0); i++) {
                int split_val = unique_values[i].item<int>();

                // Create masks for left and right splits
                auto left_mask = feature_values == split_val;
                auto right_mask = ~left_mask;

                int left_count = left_mask.sum().item<int>();
                int right_count = right_mask.sum().item<int>();

                // Skip if split doesn't satisfy minimum samples requirement
                if (left_count < min_samples_leaf || right_count < min_samples_leaf) {
                    continue;
                }

                // Calculate weighted impurities
                auto left_y = y.index({ left_mask });
                auto left_weights = sample_weights.index({ left_mask });
                double left_weight = left_weights.sum().item<double>();
                double left_impurity = calculateGiniImpurity(left_y, left_weights);

                auto right_y = y.index({ right_mask });
                auto right_weights = sample_weights.index({ right_mask });
                double right_weight = right_weights.sum().item<double>();
                double right_impurity = calculateGiniImpurity(right_y, right_weights);

                // Calculate impurity decrease
                double impurity_decrease = current_impurity -
                    (left_weight / total_weight * left_impurity +
                        right_weight / total_weight * right_impurity);

                // Update best split if this is better
                if (impurity_decrease > best_split.impurity_decrease) {
                    best_split.feature_index = feat_idx;
                    best_split.split_value = split_val;
                    best_split.impurity_decrease = impurity_decrease;
                    best_split.left_mask = left_mask;
                    best_split.right_mask = right_mask;
                }
            }
        }

        return best_split;
    }

    double DecisionTree::calculateGiniImpurity(
        const torch::Tensor& y,
        const torch::Tensor& sample_weights)
    {
        if (y.size(0) == 0 || sample_weights.size(0) == 0) {
            return 0.0;
        }

        if (y.size(0) != sample_weights.size(0)) {
            throw std::runtime_error("y and sample_weights must have same size");
        }

        torch::Tensor class_weights = torch::zeros({ n_classes });

        // Calculate weighted class counts
        for (int i = 0; i < y.size(0); i++) {
            int class_idx = y[i].item<int>();

            if (class_idx < 0 || class_idx >= n_classes) {
                throw std::runtime_error("Invalid class index: " + std::to_string(class_idx));
            }

            class_weights[class_idx] += sample_weights[i].item<float>();
        }

        // Normalize
        double total_weight = class_weights.sum().item<double>();
        if (total_weight == 0) return 0.0;

        class_weights /= total_weight;

        // Calculate Gini impurity: 1 - sum(p_i^2)
        double gini = 1.0;
        for (int i = 0; i < n_classes; i++) {
            double p = class_weights[i].item<double>();
            gini -= p * p;
        }

        return gini;
    }


    torch::Tensor DecisionTree::predict(torch::Tensor& X)
    {
        if (!fitted) {
            throw std::runtime_error(CLASSIFIER_NOT_FITTED);
        }

        int n_samples = X.size(1);
        torch::Tensor predictions = torch::zeros({ n_samples }, torch::kInt32);

        for (int i = 0; i < n_samples; i++) {
            auto sample = X.index({ torch::indexing::Slice(), i }).ravel();
            predictions[i] = predictSample(sample);
        }

        return predictions;
    }
    void dumpTensor(const torch::Tensor& tensor, const std::string& name)
    {
        std::cout << name << ": " << std::endl;
        for (int i = 0; i < tensor.size(0); i++) {
            std::cout << "[";
            for (int j = 0; j < tensor.size(1); j++) {
                std::cout << tensor[i][j].item<int>() << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    }
    void dumpVector(const std::vector<std::vector<int>>& vec, const std::string& name)
    {
        std::cout << name << ": " << std::endl;;
        for (const auto& row : vec) {
            std::cout << "[";
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << "] " << std::endl;
        }
        std::cout << std::endl;
    }

    std::vector<int> DecisionTree::predict(std::vector<std::vector<int>>& X)
    {
        // Convert to tensor
        long n = X.size();
        long m = X.at(0).size();
        torch::Tensor X_tensor = platform::TensorUtils::to_matrix(X);
        auto predictions = predict(X_tensor);
        std::vector<int> result = platform::TensorUtils::to_vector<int>(predictions);
        return result;
    }

    torch::Tensor DecisionTree::predict_proba(torch::Tensor& X)
    {
        if (!fitted) {
            throw std::runtime_error(CLASSIFIER_NOT_FITTED);
        }

        int n_samples = X.size(1);
        torch::Tensor probabilities = torch::zeros({ n_samples, n_classes });

        for (int i = 0; i < n_samples; i++) {
            auto sample = X.index({ torch::indexing::Slice(), i }).ravel();
            probabilities[i] = predictProbaSample(sample);
        }

        return probabilities;
    }

    std::vector<std::vector<double>> DecisionTree::predict_proba(std::vector<std::vector<int>>& X)
    {
        auto n_samples = X.at(0).size();
        // Convert to tensor
        torch::Tensor X_tensor = platform::TensorUtils::to_matrix(X);
        auto proba_tensor = predict_proba(X_tensor);
        std::vector<std::vector<double>> result(n_samples, std::vector<double>(n_classes, 0.0));

        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_classes; j++) {
                result[i][j] = proba_tensor[i][j].item<double>();
            }
        }

        return result;
    }

    int DecisionTree::predictSample(const torch::Tensor& x) const
    {
        if (!fitted) {
            throw std::runtime_error(CLASSIFIER_NOT_FITTED);
        }

        if (x.size(0) != n) {  // n debería ser el número de características
            throw std::runtime_error("Input sample has wrong number of features");
        }

        const TreeNode* leaf = traverseTree(x, root.get());
        return leaf->predicted_class;
    }
    torch::Tensor DecisionTree::predictProbaSample(const torch::Tensor& x) const
    {
        const TreeNode* leaf = traverseTree(x, root.get());
        return leaf->class_probabilities.clone();
    }


    const TreeNode* DecisionTree::traverseTree(const torch::Tensor& x, const TreeNode* node) const
    {
        if (!node) {
            throw std::runtime_error("Null node encountered during tree traversal");
        }

        if (node->is_leaf) {
            return node;
        }

        if (node->split_feature < 0 || node->split_feature >= x.size(0)) {
            throw std::runtime_error("Invalid split_feature index: " + std::to_string(node->split_feature));
        }

        int feature_value = x[node->split_feature].item<int>();

        if (feature_value == node->split_value) {
            if (!node->left) {
                throw std::runtime_error("Missing left child in tree");
            }
            return traverseTree(x, node->left.get());
        } else {
            if (!node->right) {
                throw std::runtime_error("Missing right child in tree");
            }
            return traverseTree(x, node->right.get());
        }
    }

    std::vector<std::string> DecisionTree::graph(const std::string& title) const
    {
        std::vector<std::string> lines;
        lines.push_back("digraph DecisionTree {");
        lines.push_back("    rankdir=TB;");
        lines.push_back("    node [shape=box, style=\"filled, rounded\", fontname=\"helvetica\"];");
        lines.push_back("    edge [fontname=\"helvetica\"];");

        if (!title.empty()) {
            lines.push_back("    label=\"" + title + "\";");
            lines.push_back("    labelloc=t;");
        }

        if (root) {
            int node_id = 0;
            treeToGraph(root.get(), lines, node_id);
        }

        lines.push_back("}");
        return lines;
    }

    void DecisionTree::treeToGraph(
        const TreeNode* node,
        std::vector<std::string>& lines,
        int& node_id,
        int parent_id,
        const std::string& edge_label) const
    {

        int current_id = node_id++;
        std::stringstream ss;

        if (node->is_leaf) {
            // Leaf node
            ss << "    node" << current_id << " [label=\"Class: " << node->predicted_class;
            ss << "\\nProb: " << std::fixed << std::setprecision(3)
                << node->class_probabilities[node->predicted_class].item<float>();
            ss << "\", fillcolor=\"lightblue\"];";
            lines.push_back(ss.str());
        } else {
            // Internal node
            ss << "    node" << current_id << " [label=\"" << features[node->split_feature];
            ss << " = " << node->split_value << "?\", fillcolor=\"lightgreen\"];";
            lines.push_back(ss.str());
        }

        // Add edge from parent
        if (parent_id >= 0) {
            ss.str("");
            ss << "    node" << parent_id << " -> node" << current_id;
            if (!edge_label.empty()) {
                ss << " [label=\"" << edge_label << "\"];";
            } else {
                ss << ";";
            }
            lines.push_back(ss.str());
        }

        // Recurse on children
        if (!node->is_leaf) {
            if (node->left) {
                treeToGraph(node->left.get(), lines, node_id, current_id, "Yes");
            }
            if (node->right) {
                treeToGraph(node->right.get(), lines, node_id, current_id, "No");
            }
        }
    }

} // namespace bayesnet