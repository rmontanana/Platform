#include <filesystem>
#include <tuple>
#include "common/Colors.h"
#include "common/CLocale.h"
#include "common/Paths.h"
#include "reports/ReportConsole.h"
#include "reports/ReportExcel.h"
#include "reports/ReportExcelCompared.h"
#include "CommandParser.h"
#include "ManageResults.h"

namespace platform {
    const std::string STATUS_OK = "Ok.";
    const std::string STATUS_COLOR = Colors::GREEN();
    ManageResults::ManageResults(int numFiles, const std::string& model, const std::string& score, bool complete, bool partial, bool compare) :
        numFiles{ numFiles }, complete{ complete }, partial{ partial }, compare{ compare }, didExcel(false), results(ResultsManager(model, score, complete, partial))
    {
        results.load();
        results.sortDate();
        sort_field = "Date";
        indexList = true;
        openExcel = false;
        workbook = NULL;
        if (numFiles == 0 or numFiles > results.size()) {
            this->numFiles = results.size();
        }
        paginator = Paginator(numFiles, results.size());
        page = 1;
    }
    void ManageResults::doMenu()
    {
        if (results.empty()) {
            std::cout << Colors::MAGENTA() << "No results found!" << Colors::RESET() << std::endl;
            return;
        }
        results.sortDate();
        list(STATUS_OK, STATUS_COLOR, -1, -1);
        menu();
        if (openExcel) {
            workbook_close(workbook);
        }
        if (didExcel) {
            std::cout << Colors::MAGENTA() << "Excel file created: " << Paths::excel() + Paths::excelResults() << std::endl;
        }
        std::cout << Colors::RESET() << "Done!" << std::endl;
    }
    void ManageResults::list(const std::string& status_message_init, const std::string& status_color, int index_A, int index_B)
    {
        //
        // Page info
        //
        int maxModel = results.maxModelSize();
        int maxTitle = results.maxTitleSize();
        std::vector<int> header_lengths = { 3, 10, maxModel, 10, 9, 3, 7, maxTitle };
        int maxLine = std::max(size_t(140), std::accumulate(header_lengths.begin(), header_lengths.end(), 0) + header_lengths.size() - 1);
        auto temp = ConfigLocale();
        auto [index_from, index_to] = paginator.getOffset(page);
        std::string suffix = "";
        if (complete) {
            suffix = " Only listing complete results ";
        }
        if (partial) {
            suffix = " Only listing partial results ";
        }
        std::string header = " " + std::to_string(index_to - index_from + 1) + " Results on screen of "
            + std::to_string(results.size()) + " - Page " + std::to_string(page) + " of "
            + std::to_string(paginator.getPages()) + " ";

        std::string prefix = std::string(maxLine - suffix.size() - header.size(), ' ');
        std::cout << Colors::CLRSCR() << Colors::REVERSE() << Colors::WHITE() << header << prefix << Colors::MAGENTA() << suffix << std::endl;
        //
        // Field names
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
        for (int i = index_from; i <= index_to; i++) {
            auto color = (i % 2) ? Colors::BLUE() : Colors::CYAN();
            std::cout << color << std::setw(3) << std::fixed << std::right << i << " ";
            std::cout << results.at(i).to_string(maxModel) << std::endl;
        }
        //
        // Status Area
        //
        std::stringstream oss;
        oss << " A: " << (index_A == -1 ? "<notset>" : std::to_string(index_A)) <<
            " B: " << (index_B == -1 ? "<notset>" : std::to_string(index_B)) << " ";
        int status_length = std::max(oss.str().size(), maxLine - oss.str().size());
        auto status_message = status_message_init.substr(0, status_length - 1);
        std::string status = status_message + std::string(std::max(size_t(0), status_length - status_message.size()), ' ');
        auto color = (index_A != -1 && index_B != -1) ? Colors::IGREEN() : Colors::IYELLOW();
        std::cout << color << Colors::REVERSE() << oss.str() << Colors::RESET() << Colors::WHITE()
            << Colors::REVERSE() << status_color << " " << status << Colors::IWHITE()
            << Colors::RESET() << std::endl;
    }
    bool ManageResults::confirmAction(const std::string& intent, const std::string& fileName) const
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
    std::string ManageResults::report_compared(const int index_A, const int index_B)
    {
        auto data_A = results.at(index_A).getJson();
        auto data_B = results.at(index_B).getJson();
        ReportExcelCompared reporter(data_A, data_B);
        reporter.report();
        didExcel = true;
        return results.at(index_A).getFilename() + " Vs " + results.at(index_B).getFilename();
    }
    std::string ManageResults::report(const int index, const bool excelReport)
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
            std::cout << reporter.fileReport();
            return "Reporting " + results.at(index).getFilename();
        }
    }
    void ManageResults::showIndex(const int index, const int idx)
    {
        // Show a dataset result inside a report
        auto data = results.at(index).getJson();
        std::cout << Colors::YELLOW() << "Showing " << results.at(index).getFilename() << std::endl;
        ReportConsole reporter(data, compare, idx);
        std::cout << reporter.fileReport();
    }
    std::pair<std::string, std::string> ManageResults::sortList()
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
    void ManageResults::menu()
    {
        char option;
        int index, subIndex, index_A = -1, index_B = -1;
        bool finished = false;
        std::string filename;
        // tuple<Option, digit, requires value>
        std::vector<std::tuple<std::string, char, bool>>  mainOptions = {
            {"quit", 'q', false},
            {"list", 'l', false},
            {"delete", 'd', true},
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
            {"report", 'r', true},
            {"list", 'l', false},
            {"back", 'b', false},
            {"quit", 'q', false}
        };
        auto parser = CommandParser();
        while (!finished) {
            bool parserError = true; // force the first iteration
            while (parserError) {
                if (indexList) {
                    auto [min_index, max_index] = paginator.getOffset(page);
                    std::tie(option, index, parserError) = parser.parse(Colors::IGREEN(), mainOptions, 'r', min_index, max_index);
                } else {
                    std::tie(option, subIndex, parserError) = parser.parse(Colors::IBLUE(), listOptions, 'r', 0, results.at(index).getJson()["results"].size() - 1);
                }
                if (parserError) {
                    if (indexList)
                        list(parser.getErrorMessage(), Colors::RED(), index_A, index_B);
                    else
                        showIndex(index, subIndex);
                }
            }
            switch (option) {
                case 'p':
                    if (paginator.valid(index)) {
                        page = index;
                        list(STATUS_OK, STATUS_COLOR, index_A, index_B);
                    } else {
                        list("Invalid page!", Colors::RED(), index_A, index_B);
                    }
                    break;
                case '+':
                    if (paginator.hasNext(page)) {
                        page++;
                        list(STATUS_OK, STATUS_COLOR, index_A, index_B);
                    } else {
                        list("No more pages!", Colors::RED(), index_A, index_B);
                    }
                    break;
                case '-':
                    if (paginator.hasPrev(page)) {
                        page--;
                        list(STATUS_OK, STATUS_COLOR, index_A, index_B);
                    } else {
                        list("First page already!", Colors::RED(), index_A, index_B);
                    }
                    break;
                case 'q':
                    finished = true;
                    break;
                case 'a':
                    if (index == index_B) {
                        list("A and B cannot be the same!", Colors::RED(), index_A, index_B);
                        break;
                    }
                    index_A = index;
                    list("A set to " + std::to_string(index), Colors::GREEN(), index_A, index_B);
                    break;
                case 'b':
                    if (indexList) {
                        if (index == index_A) {
                            list("A and B cannot be the same!", Colors::RED(), index_A, index_B);
                            break;
                        }
                        index_B = index;
                        list("B set to " + std::to_string(index), Colors::GREEN(), index_A, index_B);
                    } else {
                        // back to show the report
                        report(index, false);
                    }
                    break;
                case 'c':
                    if (index_A == -1 || index_B == -1) {
                        list("Need to set A and B first!", Colors::RED(), index_A, index_B);
                        break;
                    }
                    list(report_compared(index_A, index_B), Colors::GREEN(), index_A, index_B);
                    break;
                case 'l':
                    list(STATUS_OK, STATUS_COLOR, index_A, index_B);
                    indexList = true;
                    break;
                case 'd':
                    filename = results.at(index).getFilename();
                    if (!confirmAction("delete", filename)) {
                        list(filename + " not deleted!", Colors::YELLOW(), index_A, index_B);
                        break;
                    }
                    std::cout << "Deleting " << filename << std::endl;
                    results.deleteResult(index);
                    list(filename + " deleted!", Colors::RED(), index_A, index_B);
                    break;
                case 'h':
                    {
                        std::string status_message;
                        filename = results.at(index).getFilename();
                        if (!confirmAction("hide", filename)) {
                            list(filename + " not hidden!", Colors::YELLOW(), index_A, index_B);
                            break;
                        }
                        filename = results.at(index).getFilename();
                        std::cout << "Hiding " << filename << std::endl;
                        results.hideResult(index, Paths::hiddenResults());
                        status_message = filename + " hidden! (moved to " + Paths::hiddenResults() + ")";
                        list(status_message, Colors::YELLOW(), index_A, index_B);
                    }
                    break;
                case 's':
                    {
                        std::string status_message, status_color;
                        tie(status_color, status_message) = sortList();
                        list(status_message, status_color, index_A, index_B);
                    }
                    break;
                case 'r':
                    if (indexList) {
                        report(index, false);
                        indexList = false;
                    } else {
                        showIndex(index, subIndex);
                    }
                    break;
                case 'e':
                    list(report(index, true), Colors::GREEN(), index_A, index_B);
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
                            list(status_message, Colors::GREEN(), index_A, index_B);
                            break;
                        }
                        list("No title change!", Colors::YELLOW(), index_A, index_B);
                    }
                    break;
            }
        }
    }
} /* namespace platform */
