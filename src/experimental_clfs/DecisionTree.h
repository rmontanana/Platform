// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2024 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <memory>
#include <vector>
#include <map>
#include <torch/torch.h>
#include "bayesnet/classifiers/Classifier.h"

namespace bayesnet {

    // Forward declaration
    struct TreeNode;

    class DecisionTree : public Classifier {
    public:
        explicit DecisionTree(int max_depth = 3, int min_samples_split = 2, int min_samples_leaf = 1);
        virtual ~DecisionTree() = default;

        // Override graph method to show tree structure
        std::vector<std::string> graph(const std::string& title = "") const override;

        // Setters for hyperparameters
        void setMaxDepth(int depth) { max_depth = depth; checkValues(); }
        void setMinSamplesSplit(int samples) { min_samples_split = samples; checkValues(); }
        void setMinSamplesLeaf(int samples) { min_samples_leaf = samples; checkValues(); }

        // Override setHyperparameters
        void setHyperparameters(const nlohmann::json& hyperparameters) override;

        torch::Tensor predict(torch::Tensor& X) override;
        std::vector<int> predict(std::vector<std::vector<int>>& X) override;
        torch::Tensor predict_proba(torch::Tensor& X) override;
        std::vector<std::vector<double>> predict_proba(std::vector<std::vector<int>>& X);

    protected:
        void buildModel(const torch::Tensor& weights) override;
        void trainModel(const torch::Tensor& weights, const Smoothing_t smoothing) override
        {
            // Decision trees do not require training in the traditional sense
            // as they are built from the data directly.
            // This method can be used to set weights or other parameters if needed.
        }
    private:
        void checkValues();
        bool validateTensors(const torch::Tensor& X, const torch::Tensor& y, const torch::Tensor& sample_weights) const;
        // Tree hyperparameters
        int max_depth;
        int min_samples_split;
        int min_samples_leaf;
        int n_classes;  // Number of classes in the target variable

        // Root of the decision tree
        std::unique_ptr<TreeNode> root;

        // Build tree recursively
        std::unique_ptr<TreeNode> buildTree(
            const torch::Tensor& X,
            const torch::Tensor& y,
            const torch::Tensor& sample_weights,
            int current_depth
        );

        // Find best split for a node
        struct SplitInfo {
            int feature_index;
            int split_value;
            double impurity_decrease;
            torch::Tensor left_mask;
            torch::Tensor right_mask;
        };

        SplitInfo findBestSplit(
            const torch::Tensor& X,
            const torch::Tensor& y,
            const torch::Tensor& sample_weights
        );

        // Calculate weighted Gini impurity for multi-class
        double calculateGiniImpurity(
            const torch::Tensor& y,
            const torch::Tensor& sample_weights
        );

        // Make predictions for a single sample
        int predictSample(const torch::Tensor& x) const;

        // Make probabilistic predictions for a single sample
        torch::Tensor predictProbaSample(const torch::Tensor& x) const;

        // Traverse tree to find leaf node
        const TreeNode* traverseTree(const torch::Tensor& x, const TreeNode* node) const;

        // Convert tree to graph representation
        void treeToGraph(
            const TreeNode* node,
            std::vector<std::string>& lines,
            int& node_id,
            int parent_id = -1,
            const std::string& edge_label = ""
        ) const;
    };

    // Tree node structure
    struct TreeNode {
        bool is_leaf;

        // For internal nodes
        int split_feature;
        int split_value;
        std::unique_ptr<TreeNode> left;
        std::unique_ptr<TreeNode> right;

        // For leaf nodes
        int predicted_class;
        torch::Tensor class_probabilities;  // Probability for each class

        TreeNode() : is_leaf(false), split_feature(-1), split_value(-1), predicted_class(-1) {}
    };

} // namespace bayesnet

#endif // DECISION_TREE_H