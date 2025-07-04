#include <iostream>
#include "BestResultsMd.h"
#include "common/Utils.h" // compute_std

namespace platform {
    using json = nlohmann::ordered_json;
    void BestResultsMd::openMdFile(const std::string& name)
    {
        handler.open(name);
        if (!handler.is_open()) {
            std::cerr << "Error opening file " << name << std::endl;
            exit(1);
        }
    }
    void BestResultsMd::results_header(const std::vector<std::string>& models, const std::string& date)
    {
        this->models = models;
        auto file_name = Paths::tex() + Paths::md_output();
        openMdFile(file_name);
        handler << "<!-- This file has been generated by the platform program" << std::endl;
        handler << "  Date: " << date.c_str() << std::endl;
        handler << "" << std::endl;
        handler << "  Table of results" << std::endl;
        handler << "-->" << std::endl;
        handler << "| # | Dataset |";
        for (const auto& model : models) {
            handler << " " << model.c_str() << " |";
        }
        handler << std::endl;
        handler << "|--: | :--- |";
        for (const auto& model : models) {
            handler << " :---: |";
        }
        handler << std::endl;
    }
    void BestResultsMd::results_body(const std::vector<std::string>& datasets, json& table)
    {
        int i = 0;
        for (auto const& dataset : datasets) {
            // Find out max value for this dataset
            double max_value = 0;
            // Find out the max value for this dataset
            for (const auto& model : models) {
                double value;
                try {
                    value = table[model].at(dataset).at(0).get<double>();
                }
                catch (nlohmann::json_abi_v3_11_3::detail::out_of_range err) {
                    value = -1.0;
                }
                if (value > max_value) {
                    max_value = value;
                }
            }
            handler << "| " << ++i << " | " << dataset.c_str() << " | ";
            for (const auto& model : models) {
                double value = table[model].at(dataset).at(0).get<double>();
                double std_value = table[model].at(dataset).at(3).get<double>();
                const char* bold = value == max_value ? "**" : "";
                handler << bold << std::setprecision(4) << std::fixed << value << "±" << std::setprecision(3) << std_value << bold << " | ";
            }
            handler << std::endl;
        }
    }
    void BestResultsMd::results_footer(const std::map<std::string, std::vector<double>>& totals, const std::string& best_model)
    {
        handler << "| | **Average Score** | ";
        int nDatasets = totals.begin()->second.size();
        for (const auto& model : models) {
            double value = std::reduce(totals.at(model).begin(), totals.at(model).end()) / nDatasets;
            double std_value = compute_std(totals.at(model), value);
            const char* bold = model == best_model ? "**" : "";
            handler << bold << std::setprecision(4) << std::fixed << value << "±" << std::setprecision(3) << std::fixed << std_value << bold << " | ";
        }

        handler.close();
    }
    void BestResultsMd::postHoc_test(std::vector<PostHocLine>& postHocResults, const std::string& kind, const std::string& date)
    {
        auto file_name = Paths::tex() + Paths::md_post_hoc();
        openMdFile(file_name);
        handler << "<!-- This file has been generated by the platform program" << std::endl;
        handler << "  Date: " << date.c_str() << std::endl;
        handler << std::endl;
        handler << "  Post-hoc handler test" << std::endl;
        handler << "-->" << std::endl;
        handler << "Post-hoc " << kind << " test: H<sub>0</sub>: There is no significant differences between the control model and the other models." << std::endl << std::endl;
        handler << "| classifier | pvalue | rank | win | tie | loss | H<sub>0</sub> |" << std::endl;
        handler << "| :-- | --: | --: | --:| --: | --: | :--: |" << std::endl;
        bool first = true;
        for (auto const& line : postHocResults) {
            auto textStatus = !line.reject ? "**" : " ";
            if (first) {
                handler << "| " << line.model << " | - | " << std::fixed << std::setprecision(2) << line.rank << " | - | - | - |" << std::endl;
                first = false;
            } else {
                handler << "| " << line.model << " | " << textStatus << std::scientific << std::setprecision(4) << line.pvalue << textStatus << " |";
                handler << std::fixed << std::setprecision(2) << line.rank << " | " << line.wtl.win << " | " << line.wtl.tie << " | " << line.wtl.loss << " |";
                handler << (line.reject ? "rejected" : "**accepted**") << " |" << std::endl;
            }
        }
        handler << std::endl;
        handler.close();
    }
}
