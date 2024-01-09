#include "ManageResults.h"
#include "CommandParser.h"
#include <filesystem>
#include <tuple>
#include "Colors.h"
#include "CLocale.h"
#include "Paths.h"
#include "ReportConsole.h"
#include "ReportExcel.h"

namespace platform {

    ManageResults::ManageResults(int numFiles, const std::string& model, const std::string& score, bool complete, bool partial, bool compare) :
        numFiles{ numFiles }, complete{ complete }, partial{ partial }, compare{ compare }, results(Results(Paths::results(), model, score, complete, partial))
    {
        indexList = true;
        openExcel = false;
        workbook = NULL;
        if (numFiles == 0) {
            this->numFiles = results.size();
        }
    }
    void ManageResults::doMenu()
    {
        if (results.empty()) {
            std::cout << Colors::MAGENTA() << "No results found!" << Colors::RESET() << std::endl;
            return;
        }
        results.sortDate();
        list();
        menu();
        if (openExcel) {
            workbook_close(workbook);
        }
        std::cout << Colors::RESET() << "Done!" << std::endl;
    }
    void ManageResults::list()
    {
        auto temp = ConfigLocale();
        std::string suffix = numFiles != results.size() ? " of " + std::to_string(results.size()) : "";
        std::stringstream oss;
        oss << "Results on screen: " << numFiles << suffix;
        std::cout << Colors::GREEN() << oss.str() << std::endl;
        std::cout << std::string(oss.str().size(), '-') << std::endl;
        if (complete) {
            std::cout << Colors::MAGENTA() << "Only listing complete results" << std::endl;
        }
        if (partial) {
            std::cout << Colors::MAGENTA() << "Only listing partial results" << std::endl;
        }
        auto i = 0;
        int maxModel = results.maxModelSize();
        std::cout << Colors::GREEN() << " #  Date       " << std::setw(maxModel) << std::left << "Model" << " Score Name  Score       C/P Duration  Title" << std::endl;
        std::cout << "=== ========== " << std::string(maxModel, '=') << " =========== =========== === ========= =============================================================" << std::endl;
        bool odd = true;
        for (auto& result : results) {
            auto color = odd ? Colors::BLUE() : Colors::CYAN();
            std::cout << color << std::setw(3) << std::fixed << std::right << i++ << " ";
            std::cout << result.to_string(maxModel) << std::endl;
            if (i == numFiles) {
                break;
            }
            odd = !odd;
        }
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
    void ManageResults::report(const int index, const bool excelReport)
    {
        std::cout << Colors::YELLOW() << "Reporting " << results.at(index).getFilename() << std::endl;
        auto data = results.at(index).load();
        if (excelReport) {
            ReportExcel reporter(data, compare, workbook);
            reporter.show();
            openExcel = true;
            workbook = reporter.getWorkbook();
            std::cout << "Adding sheet to " << Paths::excel() + Paths::excelResults() << std::endl;
        } else {
            ReportConsole reporter(data, compare);
            reporter.show();
        }
    }
    void ManageResults::showIndex(const int index, const int idx)
    {
        // Show a dataset result inside a report
        auto data = results.at(index).load();
        std::cout << Colors::YELLOW() << "Showing " << results.at(index).getFilename() << std::endl;
        ReportConsole reporter(data, compare, idx);
        reporter.show();
    }
    void ManageResults::sortList()
    {
        std::cout << Colors::YELLOW() << "Choose sorting field (date='d', score='s', duration='u', model='m'): ";
        std::string line;
        char option;
        getline(std::cin, line);
        if (line.size() == 0)
            return;
        if (line.size() > 1) {
            std::cout << "Invalid option" << std::endl;
            return;
        }
        option = line[0];
        switch (option) {
            case 'd':
                results.sortDate();
                break;
            case 's':
                results.sortScore();
                break;
            case 'u':
                results.sortDuration();
                break;
            case 'm':
                results.sortModel();
                break;
            default:
                std::cout << "Invalid option" << std::endl;
        }
    }
    void ManageResults::menu()
    {
        char option;
        int index, subIndex;
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
            {"excel", 'e', true}
        };
        std::vector<std::tuple<std::string, char, bool>> listOptions = {
            {"report", 'r', true},
            {"list", 'l', false},
            {"quit", 'q', false}
        };
        auto parser = CommandParser();
        while (!finished) {
            if (indexList) {
                std::tie(option, index) = parser.parse(Colors::GREEN(), mainOptions, 'r', numFiles - 1);
            } else {
                std::tie(option, subIndex) = parser.parse(Colors::MAGENTA(), listOptions, 'r', results.at(index).load()["results"].size() - 1);
            }
            switch (option) {
                case 'q':
                    finished = true;
                    break;
                case 'l':
                    list();
                    indexList = true;
                    break;
                case 'd':
                    filename = results.at(index).getFilename();
                    if (!confirmAction("delete", filename))
                        break;
                    std::cout << "Deleting " << filename << std::endl;
                    results.deleteResult(index);
                    std::cout << "File: " + filename + " deleted!" << std::endl;
                    list();
                    break;
                case 'h':
                    filename = results.at(index).getFilename();
                    if (!confirmAction("hide", filename))
                        break;
                    filename = results.at(index).getFilename();
                    std::cout << "Hiding " << filename << std::endl;
                    results.hideResult(index, Paths::hiddenResults());
                    std::cout << "File: " + filename + " hidden! (moved to " << Paths::hiddenResults() << ")" << std::endl;
                    list();
                    break;
                case 's':
                    sortList();
                    list();
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
                    report(index, true);
                    break;
            }
        }
    }
} /* namespace platform */
