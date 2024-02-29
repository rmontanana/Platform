#include <sstream>
#include "DatasetsExcel.h"
#include "Paths.h"


namespace platform {
    DatasetsExcel::DatasetsExcel(json& data) : data(data), ExcelFile()
    {
        file_name = "datasets.xlsx";
        workbook = workbook_new(getFileName().c_str());
        createFormats();
        setProperties("Datasets");
    }
    DatasetsExcel::~DatasetsExcel()
    {
        workbook_close(workbook);
    }
    void DatasetsExcel::report()
    {
        int datasetNameSize = 25; // Min size of the column
        int balanceSize = 75; // Min size of the column
        worksheet = workbook_add_worksheet(workbook, "Datasets");
        worksheet_merge_range(worksheet, 0, 0, 0, 5, "Datasets", styles["headerFirst"]);
        formatColumns(datasetNameSize, balanceSize);
        // Body header
        row = 2;
        int col = 0;
        int i = 0;
        for (const auto& name : { "Nº", "Dataset", "Samples", "Features", "Classes", "Balance" }) {
            writeString(row, col++, name, "bodyHeader");
        }
        for (auto& [key, value] : data.items()) {
            row++;
            if (key.size() > datasetNameSize) {
                datasetNameSize = key.size();
            }
            writeInt(row, 0, i++, "ints");
            writeString(row, 1, key.c_str(), "text");
            writeInt(row, 2, value["samples"], "ints");
            writeInt(row, 3, value["features"], "ints");
            writeInt(row, 4, value["classes"], "ints");
            writeString(row, 5, value["balance"].get<std::string>().c_str(), "text");
        }
        row++;
        formatColumns(datasetNameSize, balanceSize);
    }
    void DatasetsExcel::formatColumns(int dataset, int balance)
    {
        worksheet_freeze_panes(worksheet, 4, 2);
        std::vector<int> columns_sizes = { 5, dataset, 10, 10, 10, balance };
        for (int i = 0; i < columns_sizes.size(); ++i) {
            worksheet_set_column(worksheet, i, i, columns_sizes.at(i), NULL);
        }
    }
}