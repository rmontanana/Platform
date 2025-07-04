#include <filesystem>
#include <tuple>
#include <string>
#include <algorithm>
#include "folding.hpp"
#include "common/CLocale.h"
#include "common/Paths.h"
#include "OptionsMenu.h"
#include "ManageScreen.h"
#include "reports/DatasetsConsole.h"
#include "reports/ReportConsole.h"
#include "reports/ReportExcel.h"
#include "reports/ReportExcelCompared.h"
#include <bayesnet/classifiers/TAN.h>
#include <fimdlp/CPPFImdlp.h>

namespace platform {
    const std::string STATUS_OK = "Ok.";
    const std::string STATUS_COLOR = Colors::GREEN();

    ManageScreen::ManageScreen(const std::string path_, int rows, int cols, const std::string& model, const std::string& score, const std::string& platform, bool complete, bool partial, bool compare) :
        path{ path_ }, rows{ rows }, cols{ cols }, complete{ complete }, partial{ partial }, compare{ compare }, didExcel(false), results(ResultsManager(path_, model, score, platform, complete, partial))
    {
        results.load();
        openExcel = false;
        workbook = NULL;
        maxModel = results.maxModelSize();
        maxTitle = results.maxTitleSize();
        header_lengths = { 3, 10, maxModel, 11, 10, 12, 2, 3, 7, maxTitle };
        header_labels = { " #", "Date", "Model", "Score Name", "Score", "Platform", "SD", "C/P", "Time", "Title" };
        sort_fields = { "Date", "Model", "Score", "Time" };
        updateSize(rows, cols);
        // Initializes the paginator for each output type (experiments, datasets, result)
        for (int i = 0; i < static_cast<int>(OutputType::Count); i++) {
            paginator.push_back(Paginator(this->rows, results.size()));
        }
        index_A = -1;
        index_B = -1;
        index = -1;
        subIndex = -1;
        output_type = OutputType::EXPERIMENTS;
    }
    void ManageScreen::computeSizes()
    {
        int minTitle = 10;
        // set 10 chars as minimum for Title
        auto header_title = header_lengths[header_lengths.size() - 1];
        min_columns = std::accumulate(header_lengths.begin(), header_lengths.end(), 0) + header_lengths.size() - header_title + minTitle;
        maxTitle = minTitle + cols - min_columns;
        header_lengths[header_lengths.size() - 1] = maxTitle;
        cols = std::min(cols, min_columns + maxTitle);
        for (auto& paginator_ : paginator) {
            paginator_.setPageSize(rows);
        }
    }
    bool ManageScreen::checkWrongColumns()
    {
        if (min_columns > cols) {
            std::cerr << Colors::MAGENTA() << "Make screen bigger to fit the results! " + std::to_string(min_columns - cols) + " columns needed! " << std::endl;
            return true;
        }
        return false;
    }
    void ManageScreen::updateSize(int rows_, int cols_)
    {
        rows = std::max(6, rows_ - 6); // 6 is the number of lines used by the menu & header
        cols = cols_;
        computeSizes();
    }
    void ManageScreen::doMenu()
    {
        if (results.empty()) {
            std::cerr << Colors::MAGENTA() << "No results found!" << Colors::RESET() << std::endl;
            return;
        }
        if (checkWrongColumns())
            return;
        results.sortResults(sort_field, sort_type);
        list(STATUS_OK, STATUS_COLOR);
        menu();
        if (openExcel) {
            workbook_close(workbook);
        }
        if (didExcel) {
            excelFileName = Paths::excel() + Paths::excelResults();
            std::cout << Colors::MAGENTA() << "Excel file created: " << excelFileName << std::endl;
        }
        std::cout << Colors::RESET() << "Done!" << std::endl;
    }

    std::string ManageScreen::getVersions()
    {
        std::string kfold_version = folding::KFold(5, 100).version();
        std::string bayesnet_version = bayesnet::TAN().getVersion();
        std::string mdlp_version = mdlp::CPPFImdlp::version();
        return " BayesNet: " + bayesnet_version + " Folding: " + kfold_version + " MDLP: " + mdlp_version + " ";
    }
    void ManageScreen::header()
    {
        auto [index_from, index_to] = paginator[static_cast<int>(output_type)].getOffset();
        std::string suffix = "";
        if (complete) {
            suffix = " Only listing complete results ";
        }
        if (partial) {
            suffix = " Only listing partial results ";
        }
        auto page = paginator[static_cast<int>(output_type)].getPage();
        auto pages = paginator[static_cast<int>(output_type)].getPages();
        auto lines = paginator[static_cast<int>(output_type)].getLines();
        auto total = paginator[static_cast<int>(output_type)].getTotal();
        std::string header = " Lines " + std::to_string(lines) + " of "
            + std::to_string(total) + " - Page " + std::to_string(page) + " of "
            + std::to_string(pages) + " ";
        std::string versions = getVersions();
        int filler = std::max(cols - versions.size() - suffix.size() - header.size(), size_t(0));
        std::string prefix = std::string(filler, ' ');
        std::cout << Colors::CLRSCR() << Colors::REVERSE() << Colors::WHITE() << header
            << prefix << Colors::GREEN() << versions << Colors::MAGENTA() << suffix << Colors::RESET() << std::endl;
    }
    void ManageScreen::footer(const std::string& status, const std::string& status_color)
    {
        std::stringstream oss;
        oss << " A: " << (index_A == -1 ? "<notset>" : std::to_string(index_A)) <<
            " B: " << (index_B == -1 ? "<notset>" : std::to_string(index_B)) << " ";
        int status_length = std::max(oss.str().size(), cols - oss.str().size());
        auto status_message = status.substr(0, status_length - 1);
        std::string status_line = status_message + std::string(std::max(size_t(0), status_length - status_message.size() - 1), ' ');
        auto color = (index_A != -1 && index_B != -1) ? Colors::IGREEN() : Colors::IYELLOW();
        std::cout << color << Colors::REVERSE() << oss.str() << Colors::RESET() << Colors::WHITE()
            << Colors::REVERSE() << status_color << " " << status_line << Colors::IWHITE()
            << Colors::RESET() << std::endl;
    }
    void ManageScreen::list(const std::string& status_message, const std::string& status_color)
    {
        switch (static_cast<int>(output_type)) {
            case static_cast<int>(OutputType::RESULT):
                list_result(status_message, status_color);
                break;
            case static_cast<int>(OutputType::DETAIL):
                list_detail(status_message, status_color);
                break;
            case static_cast<int>(OutputType::DATASETS):
                list_datasets(status_message, status_color);
                break;
            case static_cast<int>(OutputType::EXPERIMENTS):
                list_experiments(status_message, status_color);
                break;
        }
    }
    void ManageScreen::list_result(const std::string& status_message, const std::string& status_color)
    {
        auto data = results.at(index).getJson();
        ReportConsole report(data, compare);
        auto header_text = report.getHeader();
        auto body = report.getBody();
        paginator[static_cast<int>(output_type)].setTotal(body.size());
        // We need to subtract 8 from the page size to make room for the extra header in report
        auto page_size = paginator[static_cast<int>(OutputType::EXPERIMENTS)].getPageSize();
        paginator[static_cast<int>(output_type)].setPageSize(page_size - 8);
        //
        // header
        //
        header();
        //
        // Results
        //
        std::cout << header_text;
        auto [index_from, index_to] = paginator[static_cast<int>(output_type)].getOffset();
        for (int i = index_from; i <= index_to; i++) {
            std::cout << body[i];
        }
        //
        // Status Area
        //
        footer(status_message, status_color);
    }
    void ManageScreen::list_detail(const std::string& status_message, const std::string& status_color)
    {
        auto data = results.at(index).getJson();
        ReportConsole report(data, compare, subIndex);
        auto header_text = report.getHeader();
        auto body = report.getBody();
        paginator[static_cast<int>(output_type)].setTotal(body.size());
        // We need to subtract 8 from the page size to make room for the extra header in report
        auto page_size = paginator[static_cast<int>(OutputType::EXPERIMENTS)].getPageSize();
        paginator[static_cast<int>(output_type)].setPageSize(page_size - 8);
        //
        // header
        //
        header();
        //
        // Results
        //
        std::cout << header_text;
        auto [index_from, index_to] = paginator[static_cast<int>(output_type)].getOffset();
        for (int i = index_from; i <= index_to; i++) {
            std::cout << body[i];
        }
        //
        // Status Area
        //
        footer(status_message, status_color);
    }
    void ManageScreen::list_datasets(const std::string& status_message, const std::string& status_color)
    {
        auto report = DatasetsConsole();
        report.report();
        paginator[static_cast<int>(output_type)].setTotal(report.getNumLines());
        //
        // header
        //
        header();
        //
        // Results
        //
        auto body = report.getBody();
        std::cout << report.getHeader();
        auto [index_from, index_to] = paginator[static_cast<int>(output_type)].getOffset();
        for (int i = index_from; i <= index_to; i++) {
            std::cout << body[i];
        }
        //
        // Status Area
        //
        footer(status_message, status_color);
    }
    void ManageScreen::list_experiments(const std::string& status_message, const std::string& status_color)
    {
        //
        // header
        //
        header();
        std::cout << Colors::RESET();
        std::string arrow_dn = Symbols::down_arrow + " ";
        std::string arrow_up = Symbols::up_arrow + " ";
        for (int i = 0; i < header_labels.size(); i++) {
            std::string suffix = "", color = Colors::GREEN();
            int diff = 0;
            if (header_labels[i] == sort_fields[static_cast<int>(sort_field)]) {
                color = Colors::YELLOW();
                diff = 2;
                suffix = sort_type == SortType::ASC ? arrow_up : arrow_dn;
            }
            std::cout << color << std::setw(header_lengths[i] + diff) << std::left << std::string(header_labels[i] + suffix) << " ";
        }
        std::cout << std::endl;
        for (int i = 0; i < header_labels.size(); i++) {
            std::cout << std::string(header_lengths[i], '=') << " ";
        }
        std::cout << Colors::RESET() << std::endl;
        //
        // Results
        //
        if (results.empty()) {
            std::cout << "No results found!" << std::endl;
            return;
        }
        auto [index_from, index_to] = paginator[static_cast<int>(output_type)].getOffset();
        for (int i = index_from; i <= index_to; i++) {
            auto color = (i % 2) ? Colors::BLUE() : Colors::CYAN();
            auto color_status = results.at(i).check().size() == 0 ? color : Colors::RED();
            std::cout << color_status << std::setw(3) << std::fixed << std::right << i << " ";
            std::cout << color << results.at(i).to_string(maxModel, maxTitle) << std::endl;
        }
        //
        // Status Area
        //
        footer(status_message, status_color);
    }
    bool ManageScreen::confirmAction(const std::string& intent, const std::string& fileName) const
    {
        std::string color;
        if (intent == "delete") {
            color = Colors::RED();
        } else {
            color = Colors::YELLOW();
        }
        std::string line;
        bool finished = false;
        while (!finished) {
            std::cout << color << "Really want to " << intent << " " << fileName << "? (y/n): ";
            getline(std::cin, line);
            finished = line.size() == 1 && (tolower(line[0]) == 'y' || tolower(line[0]) == 'n');
        }
        if (tolower(line[0]) == 'y') {
            return true;
        }
        std::cout << "Not done!" << std::endl;
        return false;
    }
    std::string ManageScreen::report_compared()
    {
        auto data_A = results.at(index_A).getJson();
        auto data_B = results.at(index_B).getJson();
        ReportExcelCompared reporter(data_A, data_B);
        reporter.report();
        didExcel = true;
        return results.at(index_A).getFilename() + " Vs " + results.at(index_B).getFilename();
    }
    std::string ManageScreen::report(const int index, const bool excelReport)
    {
        auto data = results.at(index).getJson();
        if (excelReport) {
            didExcel = true;
            ReportExcel reporter(data, compare, workbook);
            reporter.show();
            openExcel = true;
            workbook = reporter.getWorkbook();
            return results.at(index).getFilename() + "->" + Paths::excel() + Paths::excelResults();
        } else {
            ReportConsole reporter(data, compare);
            std::cout << Colors::CLRSCR() << reporter.fileReport();
            return "Reporting " + results.at(index).getFilename();
        }
    }
    void ManageScreen::changeModel(const int index)
    {
        std::cout << "Old model: " << results.at(index).getModel() << std::endl;
        std::cout << "New model: ";
        std::string newModel;
        getline(std::cin, newModel);
        if (newModel.empty()) {
            list("Model not changed", Colors::YELLOW());
            return;
        }
        if (newModel == results.at(index).getModel()) {
            list("Model already set to " + newModel, Colors::RED());
            return;
        }
        // Remove the old result file
        std::string oldFile = path + results.at(index).getFilename();
        std::filesystem::remove(oldFile);
        // Actually change the model
        results.at(index).setModel(newModel);
        results.at(index).save(path);
        int newModelSize = static_cast<int>(newModel.size());
        if (newModelSize > maxModel) {
            maxModel = newModelSize;
            header_lengths[2] = maxModel;
            updateSize(rows, cols);
        }
        list("Model changed to " + newModel, Colors::GREEN());
    }
    std::pair<std::string, std::string> ManageScreen::sortList()
    {
        std::vector<std::tuple<std::string, char, bool>>  sortOptions = {
            {"date", 'd', false},
            {"score", 's', false},
            {"time", 't', false},
            {"model", 'm', false},
            {"ascending+", '+', false},
            {"descending-", '-', false}
        };
        auto sortMenu = OptionsMenu(sortOptions, Colors::YELLOW(), Colors::RED(), cols);
        std::string invalid_option = "Invalid sorting option";
        char option;
        bool parserError = true; // force the first iteration
        while (parserError) {
            if (checkWrongColumns())
                return { Colors::RED(), "Invalid column size" };
            auto [min_index, max_index] = paginator[static_cast<int>(output_type)].getOffset();
            std::tie(option, index, parserError) = sortMenu.parse(' ', 0, 0);
            sortMenu.updateColumns(cols);
            if (parserError) {
                return { Colors::RED(), invalid_option };
            }
        }
        switch (option) {
            case 'd':
                sort_field = SortField::DATE;
                break;
            case 's':
                sort_field = SortField::SCORE;
                break;
            case 't':
                sort_field = SortField::DURATION;
                break;
            case 'm':
                sort_field = SortField::MODEL;
                break;
            case '+':
                sort_type = SortType::ASC;
                break;
            case '-':
                sort_type = SortType::DESC;
                break;
            default:
                return { Colors::RED(), invalid_option };
        }
        results.sortResults(sort_field, sort_type);
        return { Colors::GREEN(), "Sorted by " + sort_fields[static_cast<int>(sort_field)] + " " + (sort_type == SortType::ASC ? "ascending" : "descending") };
    }
    void ManageScreen::menu()
    {
        char option;
        bool finished = false;
        std::string filename;
        // tuple<Option, digit, requires value>
        std::vector<std::tuple<std::string, char, bool>>  mainOptions = {
            {"quit", 'q', false},
            {"list", 'l', false},
            {"Delete", 'D', true},
            {"datasets", 'd', false},
            {"change model", 'm', true},
            {"hide", 'h', true},
            {"sort", 's', false},
            {"report", 'r', true},
            {"excel", 'e', true},
            {"title", 't', true},
            {"set A", 'A', true},
            {"set B", 'B', true},
            {"compare A~B", 'c', false},
            {"page", 'p', true},
            {"Page+", '+', false },
            {"Page-", '-', false}
        };
        // tuple<Option, digit, requires value>
        std::vector<std::tuple<std::string, char, bool>> listOptions = {
            {"quit", 'q', false},
            {"report", 'r', true},
            {"list", 'l', false},
            {"excel", 'e', true},
            {"back", 'b', false},
            {"page", 'p', true},
            {"Page+", '+', false},
            {"Page-", '-', false}
        };
        while (!finished) {
            auto main_menu = OptionsMenu(mainOptions, Colors::IGREEN(), Colors::YELLOW(), cols);
            auto list_menu = OptionsMenu(listOptions, Colors::IBLUE(), Colors::YELLOW(), cols);
            OptionsMenu& menu = output_type == OutputType::EXPERIMENTS ? main_menu : list_menu;
            bool parserError = true; // force the first iteration
            while (parserError) {
                int index_menu;
                auto [min_index, max_index] = paginator[static_cast<int>(output_type)].getOffset();
                std::tie(option, index_menu, parserError) = menu.parse('r', min_index, max_index);
                if (output_type == OutputType::EXPERIMENTS) {
                    index = index_menu;
                } else {
                    subIndex = index_menu;
                }
                if (min_columns > cols) {
                    std::cerr << "Make screen bigger to fit the results! " + std::to_string(min_columns - cols) + " columns needed! " << std::endl;
                    return;
                }
                menu.updateColumns(cols);
                if (parserError) {
                    list(menu.getErrorMessage(), Colors::RED());
                }
            }
            switch (option) {
                case 'd':
                    output_type = OutputType::DATASETS;
                    list_datasets(STATUS_OK, STATUS_COLOR);
                    break;
                case 'p':
                    {
                        auto page = output_type == OutputType::EXPERIMENTS ? index : subIndex;
                        if (paginator[static_cast<int>(output_type)].setPage(page)) {
                            list(STATUS_OK, STATUS_COLOR);
                        } else {
                            list("Invalid page! (" + std::to_string(page) + ")", Colors::RED());
                        }
                    }
                    break;
                case '+':
                    if (paginator[static_cast<int>(output_type)].addPage()) {
                        list(STATUS_OK, STATUS_COLOR);
                    } else {
                        list("No more pages!", Colors::RED());
                    }
                    break;
                case '-':
                    if (paginator[static_cast<int>(output_type)].subPage()) {
                        list(STATUS_OK, STATUS_COLOR);
                    } else {
                        list("First page already!", Colors::RED());
                    }
                    break;
                case 'q':
                    finished = true;
                    break;
                case 'A':
                    if (index == index_B) {
                        list("A and B cannot be the same!", Colors::RED());
                        break;
                    }
                    index_A = index;
                    list("A set to " + std::to_string(index), Colors::GREEN());
                    break;
                case 'B': // set_b
                    if (index == index_A) {
                        list("A and B cannot be the same!", Colors::RED());
                        break;
                    }
                    index_B = index;
                    list("B set to " + std::to_string(index), Colors::GREEN());
                    break;
                case 'b': // back to list
                    // back to show the report
                    output_type = OutputType::RESULT;
                    paginator[static_cast<int>(OutputType::DETAIL)].setPage(1);
                    list(STATUS_OK, STATUS_COLOR);
                    break;
                case 'c':
                    if (index_A == -1 || index_B == -1) {
                        list("Need to set A and B first!", Colors::RED());
                        break;
                    }
                    list(report_compared(), Colors::GREEN());
                    break;
                case 'l':
                    output_type = OutputType::EXPERIMENTS;
                    paginator[static_cast<int>(OutputType::DATASETS)].setPage(1);
                    paginator[static_cast<int>(OutputType::RESULT)].setPage(1);
                    paginator[static_cast<int>(OutputType::DETAIL)].setPage(1);
                    list(STATUS_OK, STATUS_COLOR);
                    break;
                case 'D':
                    filename = results.at(index).getFilename();
                    if (!confirmAction("delete", filename)) {
                        list(filename + " not deleted!", Colors::YELLOW());
                        break;
                    }
                    std::cout << "Deleting " << filename << std::endl;
                    results.deleteResult(index);
                    paginator[static_cast<int>(OutputType::EXPERIMENTS)].setTotal(results.size());
                    list(filename + " deleted!", Colors::RED());
                    break;
                case 'm':
                    changeModel(index);
                    break;
                case 'h':
                    {
                        std::string status_message;
                        filename = results.at(index).getFilename();
                        if (!confirmAction("hide", filename)) {
                            list(filename + " not hidden!", Colors::YELLOW());
                            break;
                        }
                        filename = results.at(index).getFilename();
                        std::cout << "Hiding " << filename << std::endl;
                        results.hideResult(index, Paths::hiddenResults());
                        status_message = filename + " hidden! (moved to " + Paths::hiddenResults() + ")";
                        paginator[static_cast<int>(OutputType::EXPERIMENTS)].setTotal(results.size());
                        list(status_message, Colors::YELLOW());
                    }
                    break;
                case 's':
                    {
                        std::string status_message, status_color;
                        tie(status_color, status_message) = sortList();
                        list(status_message, status_color);
                    }
                    break;
                case 'r':
                    if (output_type == OutputType::DATASETS) {
                        list(STATUS_OK, STATUS_COLOR);
                        break;
                    }
                    if (output_type == OutputType::EXPERIMENTS) {
                        output_type = OutputType::RESULT;
                        paginator[static_cast<int>(OutputType::DETAIL)].setPage(1);
                        list(STATUS_OK, STATUS_COLOR);
                    } else {
                        output_type = OutputType::DETAIL;
                        list(STATUS_OK, STATUS_COLOR);
                    }
                    break;
                case 'e':
                    if (output_type == OutputType::EXPERIMENTS) {
                        list(report(index, true), Colors::GREEN());
                        break;
                    }
                    list(report(subIndex, true), Colors::GREEN());
                    break;
                case 't':
                    {
                        std::cout << "Title: " << results.at(index).getTitle() << std::endl;
                        std::cout << "New title: ";
                        std::string newTitle;
                        getline(std::cin, newTitle);
                        if (!newTitle.empty()) {
                            results.at(index).setTitle(newTitle);
                            results.at(index).save(path);
                            list("Title changed to " + newTitle, Colors::GREEN());
                            break;
                        }
                        list("No title change!", Colors::YELLOW());
                    }
                    break;
            }
        }
    }
} /* namespace platform */
