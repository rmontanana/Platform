#ifndef SCORES_H
#define SCORES_H
#include <vector>
#include <string>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
namespace platform {
    using json = nlohmann::ordered_json;
    class Scores {
    public:
        Scores(torch::Tensor& y_test, torch::Tensor& y_pred, int num_classes, std::vector<std::string> labels = {});
        explicit Scores(json& confusion_matrix_);
        float accuracy();
        float f1_score(int num_class);
        float f1_weighted();
        float f1_macro();
        float precision(int num_class);
        float recall(int num_class);
        torch::Tensor get_confusion_matrix() { return confusion_matrix; }
        std::vector<std::string> classification_report(std::string color = "", std::string title = "");
        json get_confusion_matrix_json(bool labels_as_keys = false);
        void aggregate(const Scores& a);
    private:
        std::string classification_report_line(std::string label, float precision, float recall, float f1_score, int support);
        void init_confusion_matrix();
        void init_default_labels();
        void compute_accuracy_value();
        int num_classes;
        float accuracy_value;
        int total;
        std::vector<std::string> labels;
        torch::Tensor confusion_matrix; // Rows ar actual, columns are predicted
        int label_len = 16;
        int dlen = 9;
        int ndec = 7;
    };
}
#endif