#include "tensor.h"

using namespace kneron::model;

size_t internal::sizeOf(const std::vector<size_t>& dims) noexcept {
    if (dims.empty())
        return 0;
    size_t size = 1;
    for (auto d : dims)
        size *= d;
    return size;
}

size_t internal::offsetOf(const std::vector<size_t>& dims,
                          const std::initializer_list<size_t>& index) noexcept
{
    assert(index.size() == dims.size());
    size_t offset = 0, dim = 1;
    auto p = index.end();
    auto q = dims.end();
    while (p != index.begin()) {
        auto i = *--p;
        auto d = *--q;
        offset += dim * i;
        dim *= d;
    }
    return offset;
}

bool internal::nextIndex(const std::vector<size_t>& dims, std::vector<size_t>& index) noexcept {
    for (auto i = dims.size(); i--; ) {
        if (++index[i] < dims[i])
            return true;
        index[i] = 0;
    }
    return false;
}
