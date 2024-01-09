#include "ArffFiles.h"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>

ArffFiles::ArffFiles() = default;

std::vector<std::string> ArffFiles::getLines() const
{
    return lines;
}

unsigned long int ArffFiles::getSize() const
{
    return lines.size();
}

std::vector<std::pair<std::string, std::string>> ArffFiles::getAttributes() const
{
    return attributes;
}

std::string ArffFiles::getClassName() const
{
    return className;
}

std::string ArffFiles::getClassType() const
{
    return classType;
}

std::vector<std::vector<float>>& ArffFiles::getX()
{
    return X;
}

std::vector<int>& ArffFiles::getY()
{
    return y;
}

void ArffFiles::loadCommon(std::string fileName)
{
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::invalid_argument("Unable to open file");
    }
    std::string line;
    std::string keyword;
    std::string attribute;
    std::string type;
    std::string type_w;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '%' || line == "\r" || line == " ") {
            continue;
        }
        if (line.find("@attribute") != std::string::npos || line.find("@ATTRIBUTE") != std::string::npos) {
            std::stringstream ss(line);
            ss >> keyword >> attribute;
            type = "";
            while (ss >> type_w)
                type += type_w + " ";
            attributes.emplace_back(trim(attribute), trim(type));
            continue;
        }
        if (line[0] == '@') {
            continue;
        }
        lines.push_back(line);
    }
    file.close();
    if (attributes.empty())
        throw std::invalid_argument("No attributes found");
}

void ArffFiles::load(const std::string& fileName, bool classLast)
{
    int labelIndex;
    loadCommon(fileName);
    if (classLast) {
        className = std::get<0>(attributes.back());
        classType = std::get<1>(attributes.back());
        attributes.pop_back();
        labelIndex = static_cast<int>(attributes.size());
    } else {
        className = std::get<0>(attributes.front());
        classType = std::get<1>(attributes.front());
        attributes.erase(attributes.begin());
        labelIndex = 0;
    }
    generateDataset(labelIndex);
}
void ArffFiles::load(const std::string& fileName, const std::string& name)
{
    int labelIndex;
    loadCommon(fileName);
    bool found = false;
    for (int i = 0; i < attributes.size(); ++i) {
        if (attributes[i].first == name) {
            className = std::get<0>(attributes[i]);
            classType = std::get<1>(attributes[i]);
            attributes.erase(attributes.begin() + i);
            labelIndex = i;
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::invalid_argument("Class name not found");
    }
    generateDataset(labelIndex);
}

void ArffFiles::generateDataset(int labelIndex)
{
    X = std::vector<std::vector<float>>(attributes.size(), std::vector<float>(lines.size()));
    auto yy = std::vector<std::string>(lines.size(), "");
    auto removeLines = std::vector<int>(); // Lines with missing values
    for (size_t i = 0; i < lines.size(); i++) {
        std::stringstream ss(lines[i]);
        std::string value;
        int pos = 0;
        int xIndex = 0;
        while (getline(ss, value, ',')) {
            if (pos++ == labelIndex) {
                yy[i] = value;
            } else {
                if (value == "?") {
                    X[xIndex++][i] = -1;
                    removeLines.push_back(i);
                } else
                    X[xIndex++][i] = stof(value);
            }
        }
    }
    for (auto i : removeLines) {
        yy.erase(yy.begin() + i);
        for (auto& x : X) {
            x.erase(x.begin() + i);
        }
    }
    y = factorize(yy);
}

std::string ArffFiles::trim(const std::string& source)
{
    std::string s(source);
    s.erase(0, s.find_first_not_of(" '\n\r\t"));
    s.erase(s.find_last_not_of(" '\n\r\t") + 1);
    return s;
}

std::vector<int> ArffFiles::factorize(const std::vector<std::string>& labels_t)
{
    std::vector<int> yy;
    yy.reserve(labels_t.size());
    std::map<std::string, int> labelMap;
    int i = 0;
    for (const std::string& label : labels_t) {
        if (labelMap.find(label) == labelMap.end()) {
            labelMap[label] = i++;
        }
        yy.push_back(labelMap[label]);
    }
    return yy;
}