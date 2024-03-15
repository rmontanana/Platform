#pragma once

#include <utility>

class Paginator {
public:
    Paginator() = default;
    Paginator(int pageSize, int total) : pageSize(pageSize), total(total)
    {
        numPages = pageSize > 0 ? (total + pageSize - 1) / pageSize : 0;
    };
    ~Paginator() = default;
    int getPageSize() const { return pageSize; }
    int getTotal() const { return total; }
    std::pair<int, int> getOffset(int page) const
    {
        if (page > numPages)
            throw std::out_of_range("page out of range");
        return { (page - 1) * pageSize, std::min(total - 1, page * pageSize - 1) };
    }
    int getPages() const { return numPages; }
    bool valid(int page) const { return page > 0 && page <= numPages; }
    bool hasPrev(int page) const { return page > 1; }
    bool hasNext(int page) const { return page < getPages(); }
private:
    int pageSize;
    int total;
    int numPages;
};