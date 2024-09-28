#include "ResultsDatasetExcel.h"
namespace platform {
    ResultsDatasetExcel::ResultsDatasetExcel()
    {
        file_name = "some_results.xlsx";
        workbook = workbook_new(getFileName().c_str());
        createFormats();
        setProperties("Results");
    }
    ResultsDatasetExcel::~ResultsDatasetExcel()
    {
        workbook_close(workbook);
    }
    void ResultsDatasetExcel::report(json& data)
    {
        worksheet = workbook_add_worksheet(workbook, data["dataset"].get<std::string>().c_str());
        // Header
        std::string title = "Results of dataset " + data["dataset"].get<std::string>() + " -  for " + data["model"].get<std::string>() + " model";
        worksheet_merge_range(worksheet, 0, 0, 0, 5, title.c_str(), styles["headerFirst"]);
        // Body header
        row = 2;
        int col = 0;
        for (const auto& name : { "#", "Model", "Date", "Time", "Score", "Hyperparameters" }) {
            writeString(row, col++, name, "bodyHeader");
        }
        // Body
        std::string bold = "_bold";
        double maxResult = data["maxResult"].get<double>();
        for (const auto& item : data["results"]) {
            row++;
            col = 0;
            std::string style = "";
            auto score = item["score"].get<double>();
            if (score == data["max_models"][item["model"].get<std::string>()]) {
                boldBlue();
                style = bold;
            }
            if (score == maxResult) {
                boldRed();
                style = bold;
            }
            writeInt(row, col++, row - 3, "ints" + style);
            writeString(row, col++, item["model"], "text" + style);
            writeString(row, col++, item["date"], "text" + style);
            writeString(row, col++, item["time"], "text" + style);
            writeDouble(row, col++, item["score"], "result" + style);
            writeString(row, col++, item["hyperparameters"].get<std::string>().c_str(), "text" + style);
        }
        // Format columns
        worksheet_freeze_panes(worksheet, 3, 2);
        auto modelSize = data["lengths"]["maxModel"].get<int>();
        auto hyperSize = data["lengths"]["maxHyper"].get<int>();
        std::vector<int> columns_sizes = { 5, modelSize + 3, 12, 9, 11, hyperSize + 10 };
        for (int i = 0; i < columns_sizes.size(); ++i) {
            worksheet_set_column(worksheet, i, i, columns_sizes.at(i), NULL);
        }
    }
}