#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <torch/torch.h>
#include <vector>

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
}
#endif
