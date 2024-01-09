#ifndef ARFFFILES_H
#define ARFFFILES_H

#include <string>
#include <vector>

class ArffFiles {
private:
    std::vector<std::string> lines;
    std::vector<std::pair<std::string, std::string>> attributes;
    std::string className;
    std::string classType;
    std::vector<std::vector<float>> X;
    std::vector<int> y;
    void generateDataset(int);
    void loadCommon(std::string);
public:
    ArffFiles();
    void load(const std::string&, bool = true);
    void load(const std::string&, const std::string&);
    std::vector<std::string> getLines() const;
    unsigned long int getSize() const;
    std::string getClassName() const;
    std::string getClassType() const;
    static std::string trim(const std::string&);
    std::vector<std::vector<float>>& getX();
    std::vector<int>& getY();
    std::vector<std::pair<std::string, std::string>> getAttributes() const;
    static std::vector<int> factorize(const std::vector<std::string>& labels_t);
};

#endif