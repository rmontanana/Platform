# AdaBoost and DecisionTree Classifier Implementation

This implementation provides both a Decision Tree classifier and a multi-class AdaBoost classifier based on the SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss) algorithm described in the paper "Multi-class AdaBoost" by Zhu et al. Implemented in C++ using <https://claude.ai>

## Components

### 1. DecisionTree Classifier

A classic decision tree implementation that:

- Supports multi-class classification
- Handles weighted samples (essential for boosting)
- Uses Gini impurity as the splitting criterion
- Works with discrete/categorical features
- Provides both class predictions and probability estimates

#### Key Features

- **Max Depth Control**: Limit tree depth to create weak learners
- **Minimum Samples**: Control minimum samples for splitting and leaf nodes
- **Weighted Training**: Properly handles sample weights for boosting
- **Visualization**: Generates DOT format graphs of the tree structure

#### Hyperparameters

- `max_depth`: Maximum depth of the tree (default: 3)
- `min_samples_split`: Minimum samples required to split a node (default: 2)
- `min_samples_leaf`: Minimum samples required in a leaf node (default: 1)

### 2. AdaBoost Classifier

A multi-class AdaBoost implementation using DecisionTree as base estimators:

- **SAMME Algorithm**: Implements the multi-class extension of AdaBoost
- **Automatic Stumps**: Uses decision stumps (max_depth=1) by default
- **Early Stopping**: Stops if base classifier performs worse than random
- **Ensemble Visualization**: Shows the weighted combination of base estimators

#### Key Features

- **Multi-class Support**: Natural extension to K classes
- **Base Estimator Control**: Configure depth of base decision trees
- **Training Monitoring**: Track training errors and estimator weights
- **Probability Estimates**: Provides class probability predictions

#### Hyperparameters

- `n_estimators`: Number of base estimators to train (default: 50)
- `base_max_depth`: Maximum depth for base decision trees (default: 1)

## Algorithm Details

The SAMME algorithm differs from binary AdaBoost in the calculation of the estimator weight (alpha):

```
α = log((1 - err) / err) + log(K - 1)
```

where `K` is the number of classes. This formula ensures that:

- When K = 2, it reduces to standard AdaBoost
- For K > 2, base classifiers only need to be better than random guessing (1/K) rather than 50%

## Usage Example

```cpp
// Create AdaBoost with decision stumps
AdaBoost ada(100, 1);  // 100 estimators, max_depth=1

// Train
ada.fit(X_train, y_train, features, className, states, Smoothing_t::NONE);

// Predict
auto predictions = ada.predict(X_test);
auto probabilities = ada.predict_proba(X_test);

// Evaluate
float accuracy = ada.score(X_test, y_test);

// Get ensemble information
auto weights = ada.getEstimatorWeights();
auto errors = ada.getTrainingErrors();
```

## Implementation Structure

```
AdaBoost (inherits from Ensemble)
    └── Uses multiple DecisionTree instances as base estimators
         └── DecisionTree (inherits from Classifier)
              └── Implements weighted Gini impurity splitting
```

## Visualization

Both classifiers support graph visualization:

- **DecisionTree**: Shows the tree structure with split conditions
- **AdaBoost**: Shows the ensemble of weighted base estimators

Generate visualizations using:

```cpp
auto graph = classifier.graph("Title");
```

## Data Format

Both classifiers expect discrete/categorical data:

- **Features**: Integer values representing categories (stored in `torch::Tensor` or `std::vector<std::vector<int>>`)
- **Labels**: Integer values representing class indices (0, 1, ..., K-1)
- **States**: Map defining possible values for each feature and the class variable
- **Sample Weights**: Optional weights for each training sample (important for boosting)

Example data setup:

```cpp
// Features matrix (n_features x n_samples)
torch::Tensor X = torch::tensor({{0, 1, 2}, {1, 0, 1}});  // 2 features, 3 samples

// Labels vector
torch::Tensor y = torch::tensor({0, 1, 0});  // 3 samples

// States definition
std::map<std::string, std::vector<int>> states;
states["feature1"] = {0, 1, 2};  // Feature 1 can take values 0, 1, or 2
states["feature2"] = {0, 1};     // Feature 2 can take values 0 or 1
states["class"] = {0, 1};        // Binary classification
```

## Notes

- The implementation handles discrete/categorical features as indicated by the int-based data structures
- Sample weights are properly propagated through the tree building process
- The DecisionTree implementation uses equality testing for splits (suitable for categorical data)
- Both classifiers support the standard fit/predict interface from the base framework

## References

- Zhu, J., Zou, H., Rosset, S., & Hastie, T. (2009). Multi-class AdaBoost. Statistics and its interface, 2(3), 349-360.
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification and Regression Trees. Wadsworth, Belmont, CA.
