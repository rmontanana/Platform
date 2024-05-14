#include <sstream>
#include "Scores.h"
#include "common/Colors.h"
namespace platform {
    Scores::Scores(torch::Tensor& y_test, torch::Tensor& y_pred, int num_classes, std::vector<std::string> labels) : num_classes(num_classes), labels(labels)
    {
        if (labels.size() == 0) {
            init_default_labels();
        }
        total = y_test.size(0);
        accuracy_value = (y_pred == y_test).sum().item<float>() / total;
        init_confusion_matrix();
        for (int i = 0; i < total; i++) {
            int actual = y_test[i].item<int>();
            int predicted = y_pred[i].item<int>();
            confusion_matrix[actual][predicted] += 1;
        }
    }
    Scores::Scores(json& confusion_matrix_)
    {
        json values;
        total = 0;
        num_classes = confusion_matrix_.size();
        init_confusion_matrix();
        int i = 0;
        for (const auto& item : confusion_matrix_.items()) {
            values = item.value();
            json key = item.key();
            if (key.is_number_integer()) {
                labels.push_back("Class " + std::to_string(key.get<int>()));
            } else {
                labels.push_back(key.get<std::string>());
            }
            for (int j = 0; j < num_classes; ++j) {
                int value_int = values[j].get<int>();
                confusion_matrix[i][j] = value_int;
                total += value_int;
            }
            i++;
        }
        compute_accuracy_value();
    }
    void Scores::compute_accuracy_value()
    {
        accuracy_value = 0;
        for (int i = 0; i < num_classes; i++) {
            accuracy_value += confusion_matrix[i][i].item<int>();
        }
        accuracy_value /= total;
        accuracy_value = std::min(accuracy_value, 1.0f);
    }
    void Scores::init_confusion_matrix()
    {
        confusion_matrix = torch::zeros({ num_classes, num_classes }, torch::kInt32);
    }
    void Scores::init_default_labels()
    {
        for (int i = 0; i < num_classes; i++) {
            labels.push_back("Class " + std::to_string(i));
        }
    }
    void Scores::aggregate(const Scores& a)
    {
        if (a.num_classes != num_classes)
            throw std::invalid_argument("The number of classes must be the same");
        confusion_matrix += a.confusion_matrix;
        total += a.total;
        compute_accuracy_value();
    }
    float Scores::accuracy()
    {
        return accuracy_value;
    }
    float Scores::f1_score(int num_class)
    {
        // Compute f1_score in a one vs rest fashion
        auto precision_value = precision(num_class);
        auto recall_value = recall(num_class);
        if (precision_value + recall_value == 0) return 0; // Avoid division by zero (0/0 = 0)
        return 2 * precision_value * recall_value / (precision_value + recall_value);
    }
    float Scores::f1_weighted()
    {
        float f1_weighted = 0;
        for (int i = 0; i < num_classes; i++) {
            f1_weighted += confusion_matrix[i].sum().item<int>() * f1_score(i);
        }
        return f1_weighted / total;
    }
    float Scores::f1_macro()
    {
        float f1_macro = 0;
        for (int i = 0; i < num_classes; i++) {
            f1_macro += f1_score(i);
        }
        return f1_macro / num_classes;
    }
    float Scores::precision(int num_class)
    {
        int tp = confusion_matrix[num_class][num_class].item<int>();
        int fp = confusion_matrix.index({ "...", num_class }).sum().item<int>() - tp;
        int fn = confusion_matrix[num_class].sum().item<int>() - tp;
        if (tp + fp == 0) return 0; // Avoid division by zero (0/0 = 0
        return float(tp) / (tp + fp);
    }
    float Scores::recall(int num_class)
    {
        int tp = confusion_matrix[num_class][num_class].item<int>();
        int fp = confusion_matrix.index({ "...", num_class }).sum().item<int>() - tp;
        int fn = confusion_matrix[num_class].sum().item<int>() - tp;
        if (tp + fn == 0) return 0; // Avoid division by zero (0/0 = 0
        return float(tp) / (tp + fn);
    }
    std::string Scores::classification_report_line(std::string label, float precision, float recall, float f1_score, int support)
    {
        std::stringstream oss;
        oss << std::right << std::setw(label_len) << label << " ";
        if (precision == 0) {
            oss << std::string(dlen, ' ') << " ";
        } else {
            oss << std::setw(dlen) << std::setprecision(ndec) << std::fixed << precision << " ";
        }
        if (recall == 0) {
            oss << std::string(dlen, ' ') << " ";
        } else {
            oss << std::setw(dlen) << std::setprecision(ndec) << std::fixed << recall << " ";
        }
        oss << std::setw(dlen) << std::setprecision(ndec) << std::fixed << f1_score << " "
            << std::setw(dlen) << std::right << support;
        return oss.str();
    }
    std::vector<std::string> Scores::classification_report(std::string color, std::string title)
    {
        std::stringstream oss;
        std::vector<std::string> report;
        for (int i = 0; i < num_classes; i++) {
            label_len = std::max(label_len, (int)labels[i].size());
        }
        report.push_back("Classification Report using " + title + " dataset");
        report.push_back("=========================================");
        oss << std::string(label_len, ' ') << " precision recall    f1-score  support";
        report.push_back(oss.str()); oss.str("");
        oss << std::string(label_len, ' ') << " ========= ========= ========= =========";
        report.push_back(oss.str()); oss.str("");
        for (int i = 0; i < num_classes; i++) {
            report.push_back(classification_report_line(labels[i], precision(i), recall(i), f1_score(i), confusion_matrix[i].sum().item<int>()));
        }
        report.push_back(" ");
        oss << classification_report_line("accuracy", 0, 0, accuracy(), total);
        report.push_back(oss.str()); oss.str("");
        float precision_avg = 0;
        float recall_avg = 0;
        float precision_wavg = 0;
        float recall_wavg = 0;
        for (int i = 0; i < num_classes; i++) {
            int support = confusion_matrix[i].sum().item<int>();
            precision_avg += precision(i);
            precision_wavg += precision(i) * support;
            recall_avg += recall(i);
            recall_wavg += recall(i) * support;
        }
        precision_wavg /= total;
        recall_wavg /= total;
        precision_avg /= num_classes;
        recall_avg /= num_classes;
        report.push_back(classification_report_line("macro avg", precision_avg, recall_avg, f1_macro(), total));
        report.push_back(classification_report_line("weighted avg", precision_wavg, recall_wavg, f1_weighted(), total));
        report.push_back("");
        report.push_back("Confusion Matrix");
        report.push_back("================");
        auto number = total > 1000 ? 4 : 3;
        for (int i = 0; i < num_classes; i++) {
            oss << std::right << std::setw(label_len) << labels[i] << " ";
            for (int j = 0; j < num_classes; j++) {
                if (i == j) oss << Colors::GREEN();
                oss << std::setw(number) << confusion_matrix[i][j].item<int>() << " ";
                if (i == j) oss << color;
            }
            report.push_back(oss.str()); oss.str("");
        }
        return report;
    }
    json Scores::get_confusion_matrix_json(bool labels_as_keys)
    {
        json j;
        for (int i = 0; i < num_classes; i++) {
            auto r_ptr = confusion_matrix[i].data_ptr<int>();
            if (labels_as_keys) {
                j[labels[i]] = std::vector<int>(r_ptr, r_ptr + num_classes);
            } else {
                j[i] = std::vector<int>(r_ptr, r_ptr + num_classes);
            }
        }
        return j;
    }
}