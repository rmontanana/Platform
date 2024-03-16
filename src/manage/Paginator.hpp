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
    int getPageSize() const { return pageSize; }
    int getLines() const
    {
        auto [start, end] = getOffset();
        return std::min(pageSize, end - start + 1);
    }
    int getTotal() const { return total; }
    void setTotal(int total) { this->total = total; computePages(); }
    std::pair<int, int> getOffset() const
    {
        return { (page - 1) * pageSize, std::min(total - 1, page * pageSize - 1) };
    }
    int getPages() const { return numPages; }
    int getPage() const { return page; }
    bool valid(int page) const { return page > 0 && page <= numPages; }
    bool hasPrev(int page) const { return page > 1; }
    bool hasNext(int page) const { return page < getPages(); }
    bool setPage(int page) { return valid(page) ? this->page = page, true : false; }
    bool addPage() { return page < numPages ? ++page, true : false; }
    bool subPage() { return page > 1 ? --page, true : false; }
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