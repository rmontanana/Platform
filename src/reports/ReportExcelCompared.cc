#include "ReportExcelCompared.h"

namespace platform {

    ReportExcelCompared::ReportExcelCompared(json& data_A, json& data_B) : data_A(data_A), data_B(data_B), ExcelFile(NULL, NULL)
    {
    }
    ReportExcelCompared::~ReportExcelCompared()
    {
        if (workbook)
            workbook_close(workbook);
    }
    void ReportExcelCompared::report()
    {
        // Create a new workbook and add the two worksheets A & B
        workbook = workbook_new((Paths::excel() + Paths::excelResults()).c_str());
        worksheet = workbook_add_worksheet(workbook, "A");
        createFormats();
        ReportExcel report(data_A, false, workbook, worksheet);
        workbook = report.getWorkbook();
        report.show();
        worksheet = workbook_add_worksheet(workbook, "B");
        report = ReportExcel(data_B, false, workbook, worksheet);
        report.show();
        // Add the comparison worksheet
        worksheet = workbook_add_worksheet(workbook, "Δ");
        header();
        body();
    }
    void ReportExcelCompared::header()
    {
        worksheet_merge_range(worksheet, 0, 0, 0, 23, "Compare Results A vs B", styles["headerFirst"]);
        worksheet_merge_range(worksheet, 1, 0, 1, 23, "Δ = (A - B) / B", styles["headerRest"]);
        worksheet_freeze_panes(worksheet, 5, 1);
    }
    double diff(double a, double b)
    {
        return (a - b) / b;
    }
    float compute_model_number(json& rA)
    {
        float result = 0;
        int num = 0;
        int models = 0;
        bool average = false;
        std::string str_models = "Number of models: ";
        for (const std::string& note : rA["notes"]) {
            std::size_t found = note.find(str_models);
            if (found != std::string::npos) {
                models += stoi(note.substr(found + str_models.size()));
                num++;
                average = true;
            }
        }
        if (average)
            result = models / num;
        return result;
    }
    void ReportExcelCompared::body()
    {
        // Body Header
        auto sizes = std::vector<int>({ 22, 10, 9, 7, 12, 12, 9, 12, 12, 9, 12, 12, 9, 12, 12, 9, 12, 12, 9, 15, 15, 9, 15, 15 });
        auto head_a = std::vector<std::string>({ "Dataset", "Samples", "Features", "Classes" });
        auto head_b = std::vector<std::string>({ "Models", "Nodes", "Edges", "States", "Score", "Time" });
        int headerRow = 3;
        int col = 0;
        for (const auto& item : head_a) {
            worksheet_merge_range(worksheet, headerRow, col, headerRow + 1, col, item.c_str(), styles["bodyHeader_even"]);
            col++;
        }
        for (const auto& item : head_b) {
            worksheet_merge_range(worksheet, headerRow, col, headerRow, col + 2, item.c_str(), styles["bodyHeader_even"]);
            writeString(headerRow + 1, col++, "A", "bodyHeader");
            writeString(headerRow + 1, col++, "B", "bodyHeader");
            writeString(headerRow + 1, col++, "Δ", "bodyHeader");
        }
        worksheet_merge_range(worksheet, headerRow, col, headerRow, col + 1, "Hyperparameters", styles["bodyHeader_even"]);
        int hypCol = col;
        writeString(headerRow + 1, hypCol, "A", "bodyHeader");
        writeString(headerRow + 1, hypCol + 1, "B", "bodyHeader");
        col = 0;
        for (const auto size : sizes) {
            worksheet_set_column(worksheet, col, col, size, NULL);
            col++;
        }
        // Body Data
        row = headerRow + 2;
        int hypSize_A = 15;
        int hypSize_B = 15;
        auto compared = std::vector<std::string>({ "models", "nodes", "leaves", "depth", "score", "time" });
        auto compared_data = std::vector<double>(compared.size(), 0.0);
        auto totals_A = std::vector<double>(compared.size(), 0.0);
        auto totals_B = std::vector<double>(compared.size(), 0.0);
        std::string hyperparameters;
        for (int i = 0; i < data_A["results"].size(); i++) {
            col = 0;
            auto& r_A = data_A["results"][i];
            auto& r_B = data_B["results"][i];
            r_A["models"] = compute_model_number(r_A);
            r_B["models"] = compute_model_number(r_B);
            for (int j = 0; j < compared.size(); j++) {
                auto key = compared[j];
                compared_data[j] = diff(r_A[key].get<double>(), r_B[key].get<double>());
                totals_A[j] += r_A[key].get<double>();
                totals_B[j] += r_B[key].get<double>();
            }
            writeString(row, col++, r_A["dataset"].get<std::string>(), "text");
            writeInt(row, col++, r_A["samples"].get<int>(), "ints");
            writeInt(row, col++, r_A["features"].get<int>(), "ints");
            writeInt(row, col++, r_A["classes"].get<int>(), "ints");
            writeDouble(row, col++, r_A["models"].get<float>(), "floats");
            writeDouble(row, col++, r_B["models"].get<float>(), "floats");
            writeDouble(row, col++, compared_data[0], "percentage");
            writeDouble(row, col++, r_A["nodes"].get<float>(), "floats");
            writeDouble(row, col++, r_B["nodes"].get<float>(), "floats");
            writeDouble(row, col++, compared_data[1], "percentage");
            writeDouble(row, col++, r_A["leaves"].get<float>(), "floats");
            writeDouble(row, col++, r_B["leaves"].get<float>(), "floats");
            writeDouble(row, col++, compared_data[2], "percentage");
            writeDouble(row, col++, r_A["depth"].get<double>(), "floats");
            writeDouble(row, col++, r_B["depth"].get<double>(), "floats");
            writeDouble(row, col++, compared_data[3], "percentage");
            writeDouble(row, col++, r_A["score"].get<double>(), "result");
            writeDouble(row, col++, r_B["score"].get<double>(), "result");
            writeDouble(row, col++, compared_data[4], "percentage");
            writeDouble(row, col++, r_A["time"].get<double>(), "time");
            writeDouble(row, col++, r_B["time"].get<double>(), "time");
            writeDouble(row, col++, compared_data[5], "percentage");
            hyperparameters = r_A["hyperparameters"].dump();
            if (hyperparameters.size() > hypSize_A) {
                hypSize_A = hyperparameters.size();
            }
            writeString(row, hypCol, hyperparameters, "text");
            hyperparameters = r_B["hyperparameters"].dump();
            if (hyperparameters.size() > hypSize_B) {
                hypSize_B = hyperparameters.size();
            }
            writeString(row, hypCol + 1, hyperparameters, "text");
            row++;
        }
        // Set the right column width of hyperparameters with the maximum length
        worksheet_set_column(worksheet, hypCol, hypCol, hypSize_A + 5, NULL);
        worksheet_set_column(worksheet, hypCol + 1, hypCol + 1, hypSize_B + 5, NULL);
        // Show totals if only one dataset is present in the result
        footer(totals_A, totals_B, row);
    }
    void ReportExcelCompared::footer(std::vector<double>& totals_A, std::vector<double>& totals_B, int row)
    {
        worksheet_merge_range(worksheet, row, 0, row, 3, "Total", styles["bodyHeader_even"]);
        auto formats = std::vector<std::string>({ "floats", "floats", "floats", "floats", "result", "result" });
        int col = 4;
        for (int i = 0; i < totals_A.size(); i++) {
            writeDouble(row, col++, totals_A[i], formats[i]);
            writeDouble(row, col++, totals_B[i], formats[i]);
            writeDouble(row, col++, diff(totals_A[i], totals_B[i]), "percentage");
        }
    }
}