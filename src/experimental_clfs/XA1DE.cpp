// ***************************************************************
// SPDX-FileCopyrightText: Copyright 2025 Ricardo Montañana Gómez
// SPDX-FileType: SOURCE
// SPDX-License-Identifier: MIT
// ***************************************************************

#include "XA1DE.h"

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
        std::vector<std::vector<int>> instances = X;
        instances.push_back(y);
        int num_instances = instances[0].size();
        int num_attributes = instances.size();
        normalize_weights(num_instances);
        std::vector<int> statesv;
        for (int i = 0; i < num_attributes; i++) {
            statesv.push_back(*max_element(instances[i].begin(), instances[i].end()) + 1);
        }
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
            // aode_.show();
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
    std::vector<std::vector<int>> to_matrix(const torch::Tensor& X)
    {
        // Ensure tensor is contiguous in memory
        auto X_contig = X.contiguous();

        // Access tensor data pointer directly
        auto data_ptr = X_contig.data_ptr<int>();

        // IF you are using int64_t as the data type, use the following line
        //auto data_ptr = X_contig.data_ptr<int64_t>();
        //std::vector<std::vector<int64_t>> data(X.size(0), std::vector<int64_t>(X.size(1)));

        // Prepare output container
        std::vector<std::vector<int>> data(X.size(0), std::vector<int>(X.size(1)));

        // Fill the 2D vector in a single loop using pointer arithmetic
        int rows = X.size(0);
        int cols = X.size(1);
        for (int i = 0; i < rows; ++i) {
            std::copy(data_ptr + i * cols, data_ptr + (i + 1) * cols, data[i].begin());
        }
        return data;
    }
    std::vector<int> to_vector(const torch::Tensor& y)
    {
        // Ensure the tensor is contiguous in memory
        auto y_contig = y.contiguous();

        // Access data pointer
        auto data_ptr = y_contig.data_ptr<int>();

        // Prepare output container
        std::vector<int> data(y.size(0));

        // Copy data efficiently
        std::copy(data_ptr, data_ptr + y.size(0), data.begin());

        return data;
    }
    XA1DE& XA1DE::fit(torch::Tensor& X, torch::Tensor& y, const std::vector<std::string>& features, const std::string& className, std::map<std::string, std::vector<int>>& states, const bayesnet::Smoothing_t smoothing)
    {
        return fit(to_matrix(X), to_vector(y), features, className, states, smoothing);
    }
}