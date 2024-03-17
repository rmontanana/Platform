#pragma once

#include <utility>

class Paginator {
public:
    Paginator() = default;
    Paginator(int pageSize, int total, int page = 1) : pageSize(pageSize), total(total), page(page)
    {
        computePages();
    };
    ~Paginator() = default;
    // Getters
    int getPageSize() const { return pageSize; }
    int getLines() const
    {
        auto [start, end] = getOffset();
        return std::min(pageSize, end - start + 1);
    }
    int getPage() const { return page; }
    int getTotal() const { return total; }
    int getPages() const { return numPages; }
    std::pair<int, int> getOffset() const
    {
        return { (page - 1) * pageSize, std::min(total - 1, page * pageSize - 1) };
    }
    // Setters
    void setTotal(int total) { this->total = total; computePages(); }
    void setPageSize(int page) { this->pageSize = page; computePages(); }
    bool setPage(int page) { return valid(page) ? this->page = page, true : false; }
    // Utils
    bool valid(int page) const { return page > 0 && page <= numPages; }
    bool hasPrev(int page) const { return page > 1; }
    bool hasNext(int page) const { return page < getPages(); }
    bool addPage() { return page < numPages ? ++page, true : false; }
    bool subPage() { return page > 1 ? --page, true : false; }
    std::string to_string() const
    {
        auto offset = getOffset();
        return "Paginator: { pageSize: " + std::to_string(pageSize) + ", total: " + std::to_string(total)
            + ", page: " + std::to_string(page) + ", numPages: " + std::to_string(numPages)
            + " Offset [" + std::to_string(offset.first) + ", " + std::to_string(offset.second) + "]}";
    }
private:
    void computePages()
    {
        numPages = pageSize > 0 ? (total + pageSize - 1) / pageSize : 0;
        if (page > numPages) {
            page = numPages;
        }
    }
    int pageSize;
    int total;
    int page;
    int numPages;
};