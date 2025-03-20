// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "ExpEnsemble.h"
#include "TensorUtils.hpp"

namespace platform {
    ExpEnsemble::ExpEnsemble() : semaphore_{ CountingSemaphore::getInstance() }, Boost(false)
    {
        validHyperparameters = {};
    }
    //
    // Parents
    //
    void ExpEnsemble::add_model(std::unique_ptr<XSpode> model)
    {
        models.push_back(std::move(model));
        n_models++;
    }
    void ExpEnsemble::remove_last_model()
    {
        models.pop_back();
        n_models--;
    }
    //
    // Predict
    //
    torch::Tensor ExpEnsemble::predict(torch::Tensor& X)
    {
        auto X_ = TensorUtils::to_matrix(X);
        torch::Tensor y = torch::tensor(predict(X_));
        return y;
    }
    torch::Tensor ExpEnsemble::predict_proba(torch::Tensor& X)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto probabilities = predict_proba(X_);
        auto n_samples = X.size(1);
        int n_classes = probabilities[0].size();
        auto y = torch::zeros({ n_samples, n_classes });
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_classes; j++) {
                y[i][j] = probabilities[i][j];
            }
        }
        return y;
    }
    float ExpEnsemble::score(torch::Tensor& X, torch::Tensor& y)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto y_ = TensorUtils::to_vector<int>(y);
        return score(X_, y_);
    }
    std::vector<std::vector<double>> ExpEnsemble::predict_proba(const std::vector<std::vector<int>>& test_data)
    {
        int test_size = test_data[0].size();
        int sample_size = test_data.size();
        auto probabilities = std::vector<std::vector<double>>(test_size, std::vector<double>(getClassNumStates()));
        int chunk_size = std::min(150, int(test_size / semaphore_.getMaxCount()) + 1);
        std::vector<std::thread> threads;
        auto worker = [&](const std::vector<std::vector<int>>& samples, int begin, int chunk, int sample_size, std::vector<std::vector<double>>& predictions) {
            std::string threadName = "(V)PWorker-" + std::to_string(begin) + "-" + std::to_string(chunk);
#if defined(__linux__)
            pthread_setname_np(pthread_self(), threadName.c_str());
#else
            pthread_setname_np(threadName.c_str());
#endif

            std::vector<int> instance(sample_size);
            for (int sample = begin; sample < begin + chunk; ++sample) {
                for (int feature = 0; feature < sample_size; ++feature) {
                    instance[feature] = samples[feature][sample];
                }
                // predictions[sample] = aode_.predict_proba(instance);
            }
            semaphore_.release();
            };
        for (int begin = 0; begin < test_size; begin += chunk_size) {
            int chunk = std::min(chunk_size, test_size - begin);
            semaphore_.acquire();
            threads.emplace_back(worker, test_data, begin, chunk, sample_size, std::ref(probabilities));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return probabilities;
    }
    std::vector<int> ExpEnsemble::predict(std::vector<std::vector<int>>& test_data)
    {
        if (!fitted) {
            throw std::logic_error(CLASSIFIER_NOT_FITTED);
        }
        auto probabilities = predict_proba(test_data);
        std::vector<int> predictions(probabilities.size(), 0);

        for (size_t i = 0; i < probabilities.size(); i++) {
            predictions[i] = std::distance(probabilities[i].begin(), std::max_element(probabilities[i].begin(), probabilities[i].end()));
        }

        return predictions;
    }
    float ExpEnsemble::score(std::vector<std::vector<int>>& test_data, std::vector<int>& labels)
    {
        Timer timer;
        timer.start();
        std::vector<int> predictions = predict(test_data);
        int correct = 0;

        for (size_t i = 0; i < predictions.size(); i++) {
            if (predictions[i] == labels[i]) {
                correct++;
            }
        }
        if (debug) {
            std::cout << "* Time to predict: " << timer.getDurationString() << std::endl;
        }
        return static_cast<float>(correct) / predictions.size();
    }

    //
    // statistics
    //
    int ExpEnsemble::getNumberOfNodes() const
    {
        if (models_.empty()) {
            return 0;
        }
        return n_models * (models_.at(0)->getNFeatures() + 1);
    }
    int ExpEnsemble::getNumberOfEdges() const
    {
        if (models_.empty()) {
            return 0;
        }
        return n_models * (2 * models_.at(0)->getNFeatures() - 1);
    }
    int ExpEnsemble::getNumberOfStates() const
    {
        if (models_.empty()) {
            return 0;
        }
        auto states = models_.at(0)->getStates();
        int nFeatures = models_.at(0)->getNFeatures();
        return std::accumulate(states.begin(), states.end(), 0) * nFeatures * n_models;
    }
    int ExpEnsemble::getClassNumStates() const
    {
        if (models_.empty()) {
            return 0;
        }
        return models_.at(0)->statesClass();
    }


}