// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "XA1DE.h"
#include "TensorUtils.hpp"

namespace platform {
    XA1DE::XA1DE() : semaphore_{ CountingSemaphore::getInstance() }
    {
        validHyperparameters = { "use_threads" };
    }
    void XA1DE::setHyperparameters(const nlohmann::json& hyperparameters_)
    {
        auto hyperparameters = hyperparameters_;
        if (hyperparameters.contains("use_threads")) {
            use_threads = hyperparameters["use_threads"].get<bool>();
            hyperparameters.erase("use_threads");
        }
        if (!hyperparameters.empty()) {
            throw std::invalid_argument("Invalid hyperparameters" + hyperparameters.dump());
        }
    }
    XA1DE& XA1DE::fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        Timer timer, timert;
        timer.start();
        timert.start();
        // debug = true;
        std::vector<std::vector<int>> instances = X;
        instances.push_back(y);
        int num_instances = instances[0].size();
        int num_attributes = instances.size();

        normalize_weights(num_instances);
        std::vector<int> statesv;
        for (int i = 0; i < num_attributes; i++) {
            statesv.push_back(*max_element(instances[i].begin(), instances[i].end()) + 1);
        }
        // std::cout << "* States: " << statesv << std::endl;
        // std::cout << "* Weights: " << weights_ << std::endl;
        // std::cout << "* Instances: " << num_instances << std::endl;
        // std::cout << "* Attributes: " << num_attributes << std::endl;
        // std::cout << "* y: " << y << std::endl;
        // std::cout << "* x shape: " << X.size() << "x" << X[0].size() << std::endl;
        // for (int i = 0; i < num_attributes - 1; i++) {
        //     std::cout << "* " << features[i] << ": " << instances[i] << std::endl;
        // }
        // std::cout << "Starting to build the model" << std::endl;
        aode_.init(statesv);
        aode_.duration_first += timer.getDuration(); timer.start();
        std::vector<int> instance;
        for (int n_instance = 0; n_instance < num_instances; n_instance++) {
            instance.clear();
            for (int feature = 0; feature < num_attributes; feature++) {
                instance.push_back(instances[feature][n_instance]);
            }
            aode_.addSample(instance, weights_[n_instance]);
        }
        aode_.duration_second += timer.getDuration(); timer.start();
        // if (debug) aode_.show();
        aode_.computeProbabilities();
        aode_.duration_third += timer.getDuration();
        if (debug) {
            // std::cout << "* Checking coherence... ";
            // aode_.checkCoherenceApprox(1e-6);
            // std::cout << "Ok!" << std::endl;
            aode_.show();
            // std::cout << "* Accumulated first time: " << aode_.duration_first << std::endl;
            // std::cout << "* Accumulated second time: " << aode_.duration_second << std::endl;
            // std::cout << "* Accumulated third time: " << aode_.duration_third << std::endl;
            std::cout << "* Time to build the model: " << timert.getDuration() << " seconds" << std::endl;
            // exit(1);
        }
        fitted = true;
        return *this;
    }
    std::vector<std::vector<double>> XA1DE::predict_proba(std::vector<std::vector<int>>& test_data)
    {
        if (use_threads) {
            return predict_proba_threads(test_data);
        }
        int test_size = test_data[0].size();
        std::vector<std::vector<double>> probabilities;

        std::vector<int> instance;
        for (int i = 0; i < test_size; i++) {
            instance.clear();
            for (int j = 0; j < (int)test_data.size(); j++) {
                instance.push_back(test_data[j][i]);
            }
            probabilities.push_back(aode_.predict_proba(instance));
        }
        return probabilities;
    }
    std::vector<std::vector<double>> XA1DE::predict_proba_threads(const std::vector<std::vector<int>>& test_data)
    {
        int test_size = test_data[0].size();
        int sample_size = test_data.size();
        auto probabilities = std::vector<std::vector<double>>(test_size, std::vector<double>(aode_.statesClass()));

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
                predictions[sample] = aode_.predict_proba(instance);
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
    std::vector<int> XA1DE::predict(std::vector<std::vector<int>>& test_data)
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
    float XA1DE::score(std::vector<std::vector<int>>& test_data, std::vector<int>& labels)
    {
        aode_.duration_first = 0.0;
        aode_.duration_second = 0.0;
        aode_.duration_third = 0.0;
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
            std::cout << "* Accumulated first time: " << aode_.duration_first << std::endl;
            std::cout << "* Accumulated second time: " << aode_.duration_second << std::endl;
            std::cout << "* Accumulated third time: " << aode_.duration_third << std::endl;
        }
        return static_cast<float>(correct) / predictions.size();
    }

    //
    // statistics
    //
    int XA1DE::getNumberOfNodes() const
    {
        return aode_.getNumberOfNodes();
    }
    int XA1DE::getNumberOfEdges() const
    {
        return aode_.getNumberOfEdges();
    }
    int XA1DE::getNumberOfStates() const
    {
        return aode_.getNumberOfStates();
    }
    int XA1DE::getClassNumStates() const
    {
        return aode_.statesClass();
    }

    //
    // Fit
    //
    // fit(std::vector<std::vector<int>>& X, std::vector<int>& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    XA1DE& XA1DE::fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto y_ = TensorUtils::to_vector<int>(y);
        return fit(X_, y_, features, className, states, smoothing);
    }
    XA1DE& XA1DE::fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        torch::Tensor y = dataset[dataset.size(0) - 1];
        torch::Tensor X = dataset.slice(0, 0, dataset.size(0) - 1);
        return fit(X, y, features, className, states, smoothing);
    }
    XA1DE& XA1DE::fit(torch::Tensor& dataset, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const torch::Tensor& weights, const bayesnet::Smoothing_t smoothing)
    {
        weights_ = TensorUtils::to_vector<double>(weights);
        return fit(dataset, features, className, states, smoothing);
    }
    //
    // Predict
    //
    std::vector<int> XA1DE::predict_spode(std::vector<std::vector<int>>& test_data, int parent)
    {
        int test_size = test_data[0].size();
        int sample_size = test_data.size();
        auto predictions = std::vector<int>(test_size);

        int chunk_size = std::min(150, int(test_size / semaphore_.getMaxCount()) + 1);
        std::vector<std::thread> threads;
        auto worker = [&](const std::vector<std::vector<int>>& samples, int begin, int chunk, int sample_size, std::vector<int>& predictions) {
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
                predictions[sample] = aode_.predict_spode(instance, parent);
            }
            semaphore_.release();
            };
        for (int begin = 0; begin < test_size; begin += chunk_size) {
            int chunk = std::min(chunk_size, test_size - begin);
            semaphore_.acquire();
            threads.emplace_back(worker, test_data, begin, chunk, sample_size, std::ref(predictions));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        return predictions;
    }
    torch::Tensor XA1DE::predict(torch::Tensor& X)
    {
        auto X_ = TensorUtils::to_matrix(X);
        torch::Tensor y = torch::tensor(predict(X_));
        return y;
    }
    torch::Tensor XA1DE::predict_proba(torch::Tensor& X)
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
    float XA1DE::score(torch::Tensor& X, torch::Tensor& y)
    {
        auto X_ = TensorUtils::to_matrix(X);
        auto y_ = TensorUtils::to_vector<int>(y);
        return score(X_, y_);
    }

}