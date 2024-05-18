#include <sstream>
#include <locale>
#include "best/BestScore.h"
#include "ReportExcel.h"
namespace platform {

    ReportExcel::ReportExcel(json data_, bool compare, lxw_workbook* workbook, lxw_worksheet* worksheet) : ReportBase(data_, compare), ExcelFile(workbook, worksheet)
    {
        createFile();
        createFormats();
    }

    void ReportExcel::formatColumns()
    {
        worksheet_freeze_panes(worksheet, 6, 1);
        std::vector<int> columns_sizes = { 22, 10, 9, 7, 12, 12, 12, 12, 12, 3, 15, 12, 23 };
        for (int i = 0; i < columns_sizes.size(); ++i) {
            worksheet_set_column(worksheet, i, i, columns_sizes.at(i), NULL);
        }
    }
    void ReportExcel::createWorksheet()
    {
        const std::string name = data["model"].get<std::string>();
        std::string suffix = "";
        std::string efectiveName;
        int num = 1;
        // Create a sheet with the name of the model
        while (true) {
            efectiveName = name + suffix;
            if (workbook_get_worksheet_by_name(workbook, efectiveName.c_str())) {
                suffix = std::to_string(++num);
            } else {
                worksheet = workbook_add_worksheet(workbook, efectiveName.c_str());
                break;
            }
            if (num > 100) {
                throw std::invalid_argument("Couldn't create sheet " + efectiveName);
            }
        }
    }

    void ReportExcel::createFile()
    {
        if (workbook == NULL) {
            workbook = workbook_new((Paths::excel() + Paths::excelResults()).c_str());
        }
        if (worksheet == NULL) {
            createWorksheet();
        }
        setProperties(data["title"].get<std::string>());
        formatColumns();
    }

    void ReportExcel::closeFile()
    {
        workbook_close(workbook);
    }

    void ReportExcel::header()
    {
        auto loc = std::locale("es_ES");
        std::cout.imbue(loc);
        std::stringstream oss;
        std::string message = data["model"].get<std::string>() + " ver. " + data["version"].get<std::string>() + " " +
            data["language"].get<std::string>() + " ver. " + data["language_version"].get<std::string>() +
            " with " + std::to_string(data["folds"].get<int>()) + " Folds cross validation and " + std::to_string(data["seeds"].size()) +
            " random seeds. " + data["date"].get<std::string>() + " " + data["time"].get<std::string>();
        worksheet_merge_range(worksheet, 0, 0, 0, 12, message.c_str(), styles["headerFirst"]);
        worksheet_merge_range(worksheet, 1, 0, 1, 12, data["title"].get<std::string>().c_str(), styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 0, 3, 0, ("Score is " + data["score_name"].get<std::string>()).c_str(), styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 1, 3, 3, "Execution time", styles["headerRest"]);
        oss << std::setprecision(2) << std::fixed << data["duration"].get<float>() << " s";
        worksheet_merge_range(worksheet, 2, 4, 2, 5, oss.str().c_str(), styles["headerRest"]);
        oss.str("");
        oss.clear();
        oss << std::setprecision(2) << std::fixed << data["duration"].get<float>() / 3600 << " h";
        worksheet_merge_range(worksheet, 3, 4, 3, 5, oss.str().c_str(), styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 6, 3, 7, "Platform", styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 8, 3, 9, data["platform"].get<std::string>().c_str(), styles["headerRest"]);
        worksheet_merge_range(worksheet, 2, 10, 2, 12, ("Random seeds: " + fromVector("seeds")).c_str(), styles["headerSmall"]);
        oss.str("");
        oss.clear();
        oss << "Stratified: " << (data["stratified"].get<bool>() ? "True" : "False");
        worksheet_merge_range(worksheet, 3, 10, 3, 11, oss.str().c_str(), styles["headerSmall"]);
        oss.str("");
        oss.clear();
        oss << "Discretized: " << (data["discretized"].get<bool>() ? "True" : "False");
        worksheet_write_string(worksheet, 3, 12, oss.str().c_str(), styles["headerSmall"]);
    }
    void ReportExcel::header_notes(int row)
    {
        writeString(row, 0, "Dataset", "bodyHeader");
        worksheet_merge_range(worksheet, row, 1, row, 6, "Note", styles["bodyHeader_even"]);
    }
    void ReportExcel::append_notes(const json& r, int row)
    {
        static bool even_note = true;
        std::string suffix;
        if (even_note) {
            even_note = false;
            suffix = "_even";
        } else {
            even_note = true;
            suffix = "_odd";
        }
        lxw_format* style = NULL;
        style = styles.at("text" + suffix);
        auto initial_row = row;
        for (const auto& note : r["notes"]) {
            worksheet_merge_range(worksheet, row, 1, row, 6, note.get<std::string>().c_str(), style);
            row++;
        }
        if (row - 1 == initial_row) {
            writeString(initial_row, 0, r["dataset"].get<std::string>(), "text");
        } else {
            worksheet_merge_range(worksheet, initial_row, 0, row - 1, 0, r["dataset"].get<std::string>().c_str(), style);
        }
    }

    void ReportExcel::body()
    {
        auto head = std::vector<std::string>(
            { "Dataset", "Samples", "Features", "Classes", "Nodes", "Edges", "States", "Score", "Score Std.", "St.", "Time",
             "Time Std.", "Hyperparameters" });
        int col = 0;
        for (const auto& item : head) {
            writeString(5, col++, item, "bodyHeader");
        }
        row = 6;
        col = 0;
        int hypSize = 22;
        json lastResult;
        double totalScore = 0.0;
        std::string hyperparameters;
        bool only_one_result = data["results"].size() == 1;
        bool first_note = true;
        int notes_row = 15 + data["results"].size();
        for (const auto& r : data["results"]) {
            writeString(row, col, r["dataset"].get<std::string>(), "text");
            writeInt(row, col + 1, r["samples"].get<int>(), "ints");
            writeInt(row, col + 2, r["features"].get<int>(), "ints");
            writeInt(row, col + 3, r["classes"].get<int>(), "ints");
            writeDouble(row, col + 4, r["nodes"].get<float>(), "floats");
            writeDouble(row, col + 5, r["leaves"].get<float>(), "floats");
            writeDouble(row, col + 6, r["depth"].get<double>(), "floats");
            writeDouble(row, col + 7, r["score"].get<double>(), "result");
            writeDouble(row, col + 8, r["score_std"].get<double>(), "result");
            const std::string status = compareResult(r["dataset"].get<std::string>(), r["score"].get<double>());
            writeString(row, col + 9, status, "textCentered");
            writeDouble(row, col + 10, r["time"].get<double>(), "time");
            writeDouble(row, col + 11, r["time_std"].get<double>(), "time");
            hyperparameters = r["hyperparameters"].dump();
            if (hyperparameters.size() > hypSize) {
                hypSize = hyperparameters.size();
            }
            writeString(row, col + 12, hyperparameters, "text");
            lastResult = r;
            totalScore += r["score"].get<double>();
            row++;
            if (!only_one_result) {
                // take care of the possible notes
                if (r.find("notes") != r.end()) {
                    if (r["notes"].size() > 0) {
                        if (first_note) {
                            first_note = false;
                            header_notes(notes_row++);
                        }
                        append_notes(r, notes_row);
                        notes_row += r["notes"].size();
                    }
                }
            }
        }
        // Set the right column width of hyperparameters with the maximum length
        worksheet_set_column(worksheet, 12, 12, hypSize + 5, NULL);
        // Show totals if only one dataset is present in the result
        if (only_one_result) {
            row++;
            if (lastResult.find("notes") != lastResult.end()) {
                if (lastResult["notes"].size() > 0) {
                    writeString(row++, 1, "Notes: ", "bodyHeader");
                    for (const auto& note : lastResult["notes"]) {
                        worksheet_merge_range(worksheet, row, 2, row, 5, note.get<std::string>().c_str(), efectiveStyle("text"));
                        row++;
                    }
                }
            }
            for (const std::string& group : { "scores_train", "scores_test", "times_train", "times_test" }) {
                row++;
                col = 1;
                writeString(row, col, group, "text");
                for (double item : lastResult[group]) {
                    std::string style = group.find("scores") != std::string::npos ? "result" : "time";
                    writeDouble(row, ++col, item, style);
                }
            }
            // Set with of columns to show those totals completely
            worksheet_set_column(worksheet, 1, 1, 12, NULL);
            for (int i = 2; i < 7; ++i) {
                // doesn't work with from col to col, so...
                worksheet_set_column(worksheet, i, i, 15, NULL);
            }
        } else {
            footer(totalScore, row);
        }
    }

    void ReportExcel::showSummary()
    {
        for (const auto& item : summary) {
            worksheet_write_string(worksheet, row + 2, 1, item.first.c_str(), styles["summaryStyle"]);
            worksheet_write_number(worksheet, row + 2, 2, item.second, styles["summaryStyle"]);
            worksheet_merge_range(worksheet, row + 2, 3, row + 2, 5, meaning.at(item.first).c_str(), styles["summaryStyle"]);
            row += 1;
        }
    }

    void ReportExcel::footer(double totalScore, int row)
    {
        showSummary();
        row += 4 + summary.size();
        auto score = data["score_name"].get<std::string>();
        auto best = BestScore::getScore(score);
        if (best.first != "") {
            worksheet_merge_range(worksheet, row, 1, row, 5, (score + " compared to " + best.first + " .:").c_str(), efectiveStyle("text"));
            writeDouble(row, 6, totalScore / best.second, "result");
        }
        if (!getExistBestFile() && compare) {
            worksheet_write_string(worksheet, row + 1, 0, "*** Best Results File not found. Couldn't compare any result!", styles["summaryStyle"]);
        }
    }
}