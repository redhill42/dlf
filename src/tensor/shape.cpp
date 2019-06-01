#include "tensor/shape.h"

using namespace dlf;

size_t Shape::size() const noexcept {
    if (m_dims.empty())
        return 0;
    size_t size = 1;
    for (auto d : m_dims)
        size *= d;
    return size;
}

size_t Shape::offset(std::initializer_list<size_t> index) const noexcept {
    assert(index.size() == m_dims.size());
    size_t offset = 0, dim = 1;
    auto p = index.end();
    auto q = m_dims.end();
    while (p != index.begin()) {
        auto i = *--p;
        auto d = *--q;
        offset += dim * i;
        dim *= d;
    }
    return offset;
}

bool Shape::next(std::vector<size_t>& index) const noexcept {
    for (auto i = m_dims.size(); i--; ) {
        if (++index[i] < m_dims[i])
            return true;
        index[i] = 0;
    }
    return false;
}

bool Shape::reshape(Shape newshape) {
    size_t newsize = 1;
    int pending = -1;

    if (newshape.rank() == 0) {
        newsize = 0;
    } else {
        for (int i = 0; i < newshape.rank(); i++) {
            if (newshape[i] == size_t(-1)) {
                if (pending != -1)
                    return false;
                pending = i;
            } else {
                newsize *= newshape[i];
            }
        }
    }

    if (pending != -1)
        newshape.m_dims[pending] = size() / newsize;
    if (size() != newshape.size())
        return false;
    m_dims = std::move(newshape.m_dims);
    return true;
}
