#include <iostream>
#include <locale>
#include "Paths.h"
#include "Colors.h"
#include "Datasets.h"

const int BALANCE_LENGTH = 75;

struct separated : numpunct<char> {
    char do_decimal_point() const { return ','; }
    char do_thousands_sep() const { return '.'; }
    std::string do_grouping() const { return "\03"; }
};

void outputBalance(const std::string& balance)
{
    auto temp = std::string(balance);
    while (temp.size() > BALANCE_LENGTH - 1) {
        auto part = temp.substr(0, BALANCE_LENGTH);
        std::cout << part << std::endl;
        std::cout << setw(48) << " ";
        temp = temp.substr(BALANCE_LENGTH);
    }
    std::cout << temp << std::endl;
}

int main(int argc, char** argv)
{
    auto data = platform::Datasets(false, platform::Paths::datasets());
    locale mylocale(std::cout.getloc(), new separated);
    locale::global(mylocale);
    std::cout.imbue(mylocale);
    std::cout << Colors::GREEN() << "Dataset                        Sampl. Feat. Cls. Balance" << std::endl;
    std::string balanceBars = std::string(BALANCE_LENGTH, '=');
    std::cout << "============================== ====== ===== === " << balanceBars << std::endl;
    bool odd = true;
    for (const auto& dataset : data.getNames()) {
        auto color = odd ? Colors::CYAN() : Colors::BLUE();
        std::cout << color << setw(30) << left << dataset << " ";
        data.loadDataset(dataset);
        auto nSamples = data.getNSamples(dataset);
        std::cout << setw(6) << right << nSamples << " ";
        std::cout << setw(5) << right << data.getFeatures(dataset).size() << " ";
        std::cout << setw(3) << right << data.getNClasses(dataset) << " ";
        std::stringstream oss;
        std::string sep = "";
        for (auto number : data.getClassesCounts(dataset)) {
            oss << sep << std::setprecision(2) << fixed << (float)number / nSamples * 100.0 << "% (" << number << ")";
            sep = " / ";
        }
        outputBalance(oss.str());
        odd = !odd;
    }
    std::cout << Colors::RESET() << std::endl;
    return 0;
}
