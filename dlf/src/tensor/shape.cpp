#include "tensor/shape.h"
#include "utility.h"

using namespace dlf;

void Shape::init(const std::vector<size_t>& extents) noexcept {
    m_dims.resize(extents.size());

    if (extents.size() == 0) {
        m_size = 0;
    } else {
        size_t size = 1;
        for (size_t i = extents.size(); i-- != 0; ) {
            assert(extents[i] != 0);
            m_dims[i].extent = extents[i];
            m_dims[i].stride = size;
            size *= extents[i];
        }
        m_size = size;
    }
}

Shape::Shape(Shape&& rhs)
    : m_dims(std::move(rhs.m_dims)),
      m_size(cxx::exchange(rhs.m_size, 0))
{
}

Shape& Shape::operator=(Shape&& rhs) {
    m_dims = std::move(rhs.m_dims);
    m_size = cxx::exchange(rhs.m_size, 0);
    return *this;
}

size_t Shape::offset(std::initializer_list<size_t> index) const noexcept {
    assert(index.size() == m_dims.size());
    size_t offset = 0, i = 0;
    for (auto a : index) {
        offset += a * m_dims[i].stride;
        ++i;
    }
    return offset;
}

bool Shape::next(std::vector<size_t>& index) const noexcept {
    for (auto i = m_dims.size(); i--; ) {
        if (++index[i] < m_dims[i].extent)
            return true;
        index[i] = 0;
    }
    return false;
}

bool Shape::reshape(std::vector<size_t> extents) noexcept {
    size_t newsize = 1;
    int pending = -1;

    if (extents.size() == 0) {
        newsize = 0;
    } else {
        for (int i = 0; i < extents.size(); i++) {
            if (extents[i] == size_t(-1)) {
                if (pending != -1)
                    return false;
                pending = i;
            } else {
                newsize *= extents[i];
            }
        }
    }

    if (pending != -1) {
        extents[pending] = size() / newsize;
        newsize *= extents[pending];
    }

    if (size() != newsize)
        return false;
    init(extents);
    return true;
}
