#include <sstream>
#include "DatasetsExcel.h"
#include "Paths.h"


namespace platform {
    DatasetsExcel::DatasetsExcel()
    {
        file_name = "datasets.xlsx";
        workbook = workbook_new(getFileName().c_str());
        setProperties("Datasets");
    }
    DatasetsExcel::~DatasetsExcel()
    {
        workbook_close(workbook);
    }
    void DatasetsExcel::report()
    {
        worksheet = workbook_add_worksheet(workbook, "Datasets");
        formatColumns();
        worksheet_merge_range(worksheet, 0, 0, 0, 4, "Datasets", styles["headerFirst"]);
        // Body header
        row = 3;
        int col = 1;
        int i = 0;
        // Get Datasets
        // auto data = platform::Datasets(false, platform::Paths::datasets());
        // auto datasets = data.getNames();
        auto datasets = std::vector<std::string>{ "iris", "wine", "digits", "breast_cancer" };
        int maxDatasetName = (*std::max_element(datasets.begin(), datasets.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })).size();
        datasetNameSize = std::max(datasetNameSize, maxDatasetName);
        writeString(row, 0, "Nº", "bodyHeader");
        writeString(row, 1, "Dataset", "bodyHeader");
        for (auto const& name : datasets) {
            row++;
            writeInt(row, 0, i++, "ints");
            writeString(row, 1, name.c_str(), "text");
        }
        row++;
        formatColumns();
    }

    void DatasetsExcel::formatColumns()
    {
        worksheet_freeze_panes(worksheet, 4, 2);
        std::vector<int> columns_sizes = { 5, datasetNameSize };
        for (int i = 0; i < columns_sizes.size(); ++i) {
            worksheet_set_column(worksheet, i, i, columns_sizes.at(i), NULL);
        }
    }
}