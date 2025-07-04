#include <iostream>
#include "BestResultsTex.h"
#include "common/Utils.h" // compute_std

namespace platform {
    using json = nlohmann::ordered_json;
    void BestResultsTex::openTexFile(const std::string& name)
    {
        handler.open(name);
        if (!handler.is_open()) {
            std::cerr << "Error opening file " << name << std::endl;
            exit(1);
        }
    }
    void BestResultsTex::results_header(const std::vector<std::string>& models, const std::string& date, bool index)
    {
        this->models = models;
        auto file_name = Paths::tex() + Paths::tex_output();
        openTexFile(file_name);
        handler << "%% This file has been generated by the platform program" << std::endl;
        handler << "%% Date: " << date.c_str() << std::endl;
        handler << "%%" << std::endl;
        handler << "%% Table of results" << std::endl;
        handler << "%%" << std::endl;
        handler << "\\begin{table}[htbp] " << std::endl;
        handler << "\\centering " << std::endl;
        handler << "\\tiny " << std::endl;
        handler << "\\renewcommand{\\arraystretch }{1.2} " << std::endl;
        handler << "\\renewcommand{\\tabcolsep }{0.07cm} " << std::endl;
        auto umetric = score;
        umetric[0] = toupper(umetric[0]);
        handler << "\\caption{" << umetric << " results(mean $\\pm$ std) for all the algorithms and datasets} " << std::endl;
        handler << "\\label{tab:results_" << score << "}" << std::endl;
        std::string header_dataset_name = index ? "r" : "l";
        handler << "\\begin{tabular} {{" << header_dataset_name << std::string(models.size(), 'c').c_str() << "}}" << std::endl;
        handler << "\\hline " << std::endl;
        handler << "" << std::endl;
        for (const auto& model : models) {
            handler << "& " << model.c_str();
        }
        handler << "\\\\" << std::endl;
        handler << "\\hline" << std::endl;
    }
    void BestResultsTex::results_body(const std::vector<std::string>& datasets, json& table, bool index)
    {
        int i = 0;
        for (auto const& dataset : datasets) {
            // Find out max value for this dataset
            double max_value = 0;
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
            if (index)
                handler << ++i << " ";
            else
                handler << dataset << " ";
            for (const auto& model : models) {
                double value = table[model].at(dataset).at(0).get<double>();
                double std_value = table[model].at(dataset).at(3).get<double>();
                const char* bold = value == max_value ? "\\bfseries" : "";
                handler << "& " << bold << std::setprecision(4) << std::fixed << value << "$\\pm$" << std::setprecision(3) << std_value;
            }
            handler << "\\\\" << std::endl;
        }
    }
    void BestResultsTex::results_footer(const std::map<std::string, std::vector<double>>& totals, const std::string& best_model)
    {
        handler << "\\hline" << std::endl;
        handler << "Average ";
        int nDatasets = totals.begin()->second.size();
        for (const auto& model : models) {
            double value = std::reduce(totals.at(model).begin(), totals.at(model).end()) / nDatasets;
            double std_value = compute_std(totals.at(model), value);
            const char* bold = model == best_model ? "\\bfseries" : "";
            handler << "& " << bold << std::setprecision(4) << std::fixed << value << "$\\pm$" << std::setprecision(3) << std::fixed << std_value;
        }
        handler << "\\\\" << std::endl;
        handler << "\\hline " << std::endl;
        handler << "\\end{tabular}" << std::endl;
        handler << "\\end{table}" << std::endl;
        handler.close();
    }
    void BestResultsTex::postHoc_test(std::vector<PostHocLine>& postHocResults, const std::string& kind, const std::string& date)
    {
        auto file_name = Paths::tex() + Paths::tex_post_hoc();
        openTexFile(file_name);
        handler << "%% This file has been generated by the platform program" << std::endl;
        handler << "%% Date: " << date.c_str() << std::endl;
        handler << "%%" << std::endl;
        handler << "%% Post-hoc " << kind << " test" << std::endl;
        handler << "%%" << std::endl;
        handler << "\\begin{table}[htbp]" << std::endl;
        handler << "\\centering" << std::endl;
        handler << "\\caption{Results of the post-hoc " << kind << " test for the mean " << score << " of the algorithms.}\\label{ tab:tests }" << std::endl;
        handler << "\\begin{tabular}{lrrrrr}" << std::endl;
        handler << "\\hline" << std::endl;
        handler << "classifier & pvalue & rank & win & tie & loss\\\\" << std::endl;
        handler << "\\hline" << std::endl;
        bool first = true;
        for (auto const& line : postHocResults) {
            auto textStatus = !line.reject ? "\\bf " : " ";
            if (first) {
                handler << line.model << " & - & " << std::fixed << std::setprecision(2) << line.rank << " & - & - & - \\\\" << std::endl;
                first = false;
            } else {
                handler << line.model << " & " << textStatus << std::scientific << std::setprecision(4) << line.pvalue << " & ";
                handler << std::fixed << std::setprecision(2) << line.rank << " & " << line.wtl.win << " & " << line.wtl.tie << " & " << line.wtl.loss << "\\\\" << std::endl;
            }
        }
        handler << "\\hline " << std::endl;
        handler << "\\end{tabular}" << std::endl;
        handler << "\\end{table}" << std::endl;
        handler.close();
    }
}
