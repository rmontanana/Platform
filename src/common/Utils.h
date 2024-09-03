#ifndef UTILS_H
#define UTILS_H
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <torch/torch.h>
namespace platform {
    template <typename T>
    std::vector<T> tensorToVector(const torch::Tensor& tensor)
    {
        torch::Tensor contig_tensor = tensor.contiguous();
        auto num_elements = contig_tensor.numel();
        const T* tensor_data = contig_tensor.data_ptr<T>();
        std::vector<T> result(tensor_data, tensor_data + num_elements);
        return result;
    }
    static std::string trim(const std::string& str)
    {
        std::string result = str;
        result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](int ch) {
            return !std::isspace(ch);
            }));
        result.erase(std::find_if(result.rbegin(), result.rend(), [](int ch) {
            return !std::isspace(ch);
            }).base(), result.end());
        return result;
    }
    static std::vector<std::string> split(const std::string& text, char delimiter)
    {
        std::vector<std::string> result;
        std::stringstream ss(text);
        std::string token;
        while (std::getline(ss, token, delimiter)) {
            result.push_back(trim(token));
        }
        return result;
    }
    inline double compute_std(std::vector<double> values, double mean)
    {
        // Compute standard devation of the values
        double sum = 0.0;
        for (const auto& value : values) {
            sum += std::pow(value - mean, 2);
        }
        double variance = sum / values.size();
        return std::sqrt(variance);
    }
    inline std::string get_date()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%Y-%m-%d");
        return oss.str();
    }
    inline std::string get_time()
    {
        time_t rawtime;
        tm* timeinfo;
        time(&rawtime);
        timeinfo = std::localtime(&rawtime);
        std::ostringstream oss;
        oss << std::put_time(timeinfo, "%H:%M:%S");
        return oss.str();
    }
}
#endif