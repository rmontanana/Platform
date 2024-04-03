#include <filesystem>
#include <tuple>
#include <string>
#include <algorithm>
#include "folding.hpp"
#include "common/Colors.h"
#include "common/CLocale.h"
#include "common/Paths.h"
#include "CommandParser.h"
#include "ManageScreen.h"
#include "reports/DatasetsConsole.h"
#include "reports/ReportConsole.h"
#include "reports/ReportExcel.h"
#include "reports/ReportExcelCompared.h"
#include <bayesnet/classifiers/TAN.h>
#include "CPPFImdlp.h"

namespace platform {
    const std::string STATUS_OK = "Ok.";
    const std::string STATUS_COLOR = Colors::GREEN();
    ManageScreen::ManageScreen(int rows, int cols, const std::string& model, const std::string& score, bool complete, bool partial, bool compare) :
        rows{ rows }, cols{ cols }, complete{ complete }, partial{ partial }, compare{ compare }, didExcel(false), results(ResultsManager(model, score, complete, partial))
    {
        results.load();
        results.sortDate();
        sort_field = "Date";
        openExcel = false;
        workbook = NULL;
        this->rows = std::max(0, rows - 6); // 6 is the number of lines used by the menu & header
        cols = std::max(cols, 140);
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
    void ManageScreen::doMenu()
    {
        if (results.empty()) {
            std::cout << Colors::MAGENTA() << "No results found!" << Colors::RESET() << std::endl;
            return;
        }
        results.sortDate();
        list(STATUS_OK, STATUS_COLOR);
        menu();
        if (openExcel) {
            workbook_close(workbook);
        }
        if (didExcel) {
            std::cout << Colors::MAGENTA() << "Excel file created: " << Paths::excel() + Paths::excelResults() << std::endl;
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
        //
        // Field names
        //
        int maxModel = results.maxModelSize();
        int maxTitle = results.maxTitleSize();
        std::vector<int> header_lengths = { 3, 10, maxModel, 10, 9, 3, 7, maxTitle };
        //
        std::cout << Colors::RESET();
        std::string arrow = Symbols::downward_arrow + " ";
        std::vector<std::string> header_labels = { " #", "Date", "Model", "Score Name", "Score", "C/P", "Time", "Title" };
        for (int i = 0; i < header_labels.size(); i++) {
            std::string suffix = "", color = Colors::GREEN();
            int diff = 0;
            if (header_labels[i] == sort_field) {
                color = Colors::YELLOW();
                diff = 2;
                suffix = arrow;
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
        auto [index_from, index_to] = paginator[static_cast<int>(output_type)].getOffset();
        for (int i = index_from; i <= index_to; i++) {
            auto color = (i % 2) ? Colors::BLUE() : Colors::CYAN();
            std::cout << color << std::setw(3) << std::fixed << std::right << i << " ";
            std::cout << results.at(i).to_string(maxModel) << std::endl;
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
            finished = line.size() == 1 && (tolower(line[0]) == 'y' || tolower(line[0] == 'n'));
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
    std::pair<std::string, std::string> ManageScreen::sortList()
    {
        std::cout << Colors::YELLOW() << "Choose sorting field (date='d', score='s', time='t', model='m'): ";
        std::string line;
        char option;
        getline(std::cin, line);
        if (line.size() == 0 || line.size() > 1) {
            return { Colors::RED(), "Invalid sorting option" };
        }
        option = line[0];
        switch (option) {
            case 'd':
                results.sortDate();
                sort_field = "Date";
                return { Colors::GREEN(), "Sorted by date" };
            case 's':
                results.sortScore();
                sort_field = "Score";
                return { Colors::GREEN(), "Sorted by score" };
            case 't':
                results.sortDuration();
                sort_field = "Time";
                return { Colors::GREEN(), "Sorted by time" };
            case 'm':
                results.sortModel();
                sort_field = "Model";
                return { Colors::GREEN(), "Sorted by model" };
            default:
                return { Colors::RED(), "Invalid sorting option" };
        }
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
            {"delete", 'D', true},
            {"datasets", 'd', false},
            {"hide", 'h', true},
            {"sort", 's', false},
            {"report", 'r', true},
            {"excel", 'e', true},
            {"title", 't', true},
            {"set A", 'a', true},
            {"set B", 'b', true},
            {"compare A~B", 'c', false},
            {"Page", 'p', true},
            {"Page+", '+', false },
            {"Page-", '-', false}
        };
        // tuple<Option, digit, requires value>
        std::vector<std::tuple<std::string, char, bool>> listOptions = {
            {"quit", 'q', false},
            {"report", 'r', true},
            {"list", 'l', false},
            {"excel", 'e', false},
            {"back", 'b', false},
            {"Page", 'p', true},
            {"Page+", '+', false},
            {"Page-", '-', false}
        };

        auto parser = CommandParser();
        while (!finished) {
            bool parserError = true; // force the first iteration
            while (parserError) {
                auto [min_index, max_index] = paginator[static_cast<int>(output_type)].getOffset();
                if (output_type == OutputType::EXPERIMENTS) {
                    std::tie(option, index, parserError) = parser.parse(Colors::IGREEN(), mainOptions, 'r', min_index, max_index);
                } else {
                    std::tie(option, subIndex, parserError) = parser.parse(Colors::IBLUE(), listOptions, 'r', min_index, max_index);
                }
                if (parserError) {
                    list(parser.getErrorMessage(), Colors::RED());
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
                case 'a':
                    if (index == index_B) {
                        list("A and B cannot be the same!", Colors::RED());
                        break;
                    }
                    index_A = index;
                    list("A set to " + std::to_string(index), Colors::GREEN());
                    break;
                case 'b': // set_b or back to list
                    if (output_type == OutputType::EXPERIMENTS) {
                        if (index == index_A) {
                            list("A and B cannot be the same!", Colors::RED());
                            break;
                        }
                        index_B = index;
                        list("B set to " + std::to_string(index), Colors::GREEN());
                    } else {
                        // back to show the report
                        output_type = OutputType::RESULT;
                        paginator[static_cast<int>(OutputType::DETAIL)].setPage(1);
                        list(STATUS_OK, STATUS_COLOR);
                    }
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
                    list(filename + " deleted!", Colors::RED());
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
                        std::string status_message;
                        std::cout << "Title: " << results.at(index).getTitle() << std::endl;
                        std::cout << "New title: ";
                        std::string newTitle;
                        getline(std::cin, newTitle);
                        if (!newTitle.empty()) {
                            results.at(index).setTitle(newTitle);
                            results.at(index).save();
                            status_message = "Title changed to " + newTitle;
                            list(status_message, Colors::GREEN());
                            break;
                        }
                        list("No title change!", Colors::YELLOW());
                    }
                    break;
            }
        }
    }
} /* namespace platform */
