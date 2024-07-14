#include <sstream>
#include <algorithm>
#include <numeric>
#include <utility>
#include "RocAuc.h"
namespace platform {
    
    double RocAuc::compute(const torch::Tensor& y_proba, const torch::Tensor& labels)
    {
        size_t nClasses = y_proba.size(1);
        // In binary classification problem there's no need to calculate the average of the AUCs
        if (nClasses == 2)
            nClasses = 1;
        size_t nSamples = y_proba.size(0);
        y_test = tensorToVector(labels);
        std::vector<double> aucScores(nClasses, 0.0);
        for (size_t classIdx = 0; classIdx < nClasses; ++classIdx) {
            scoresAndLabels.clear();
            for (size_t i = 0; i < nSamples; ++i) {
                scoresAndLabels.emplace_back(y_proba[i][classIdx].item<float>(), y_test[i] == classIdx ? 1 : 0);
            }
            aucScores[classIdx] = compute_common(nSamples, classIdx);
        }
        return std::accumulate(aucScores.begin(), aucScores.end(), 0.0) / nClasses;
    }
    double RocAuc::compute(const std::vector<std::vector<double>>& y_proba, const std::vector<int>& labels)
    {
        y_test = labels;
        size_t nClasses = y_proba[0].size();
        // In binary classification problem there's no need to calculate the average of the AUCs
        if (nClasses == 2)
            nClasses = 1;
        size_t nSamples = y_proba.size();
        std::vector<double> aucScores(nClasses, 0.0);
        for (size_t classIdx = 0; classIdx < nClasses; ++classIdx) {
            scoresAndLabels.clear();
            for (size_t i = 0; i < nSamples; ++i) {
                scoresAndLabels.emplace_back(y_proba[i][classIdx], labels[i] == classIdx ? 1 : 0);
            }
            aucScores[classIdx] = compute_common(nSamples, classIdx);
        }
        return std::accumulate(aucScores.begin(), aucScores.end(), 0.0) / nClasses;
    }
    double RocAuc::compute_common(size_t nSamples, size_t classIdx)
    {
        std::sort(scoresAndLabels.begin(), scoresAndLabels.end(), std::greater<>());
        std::vector<double> tpr, fpr;
        double tp = 0, fp = 0;
        double totalPos = std::count(y_test.begin(), y_test.end(), classIdx);
        double totalNeg = nSamples - totalPos;

        for (const auto& [score, label] : scoresAndLabels) {
            if (label == 1) {
                tp += 1;
            } else {
                fp += 1;
            }
            tpr.push_back(tp / totalPos);
            fpr.push_back(fp / totalNeg);
        }
        double auc = 0.0;
        for (size_t i = 1; i < tpr.size(); ++i) {
            auc += 0.5 * (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]);
        }
        return auc;
    }
}