#include "tensor/shape.h"
#include "utility.h"

namespace dlf {

void Shape::init(const std::vector<size_t>& extents) noexcept {
    m_dims.resize(extents.size());

    if (extents.empty()) {
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

bool Shape::is_contiguous() const noexcept {
    if (m_dims.size() != 0) {
        size_t size = 1;
        for (size_t i = m_dims.size(); i-- != 0;) {
            if (m_dims[i].stride == 0 && m_dims[i].extent == 1)
                continue;
            if (m_dims[i].stride != size)
                return false;
            size *= m_dims[i].extent;
        }
    }
    return true;
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

size_t Shape::offset(const std::vector<size_t>& index) const noexcept {
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

bool Shape::previous(std::vector<size_t>& index) const noexcept {
    for (auto i = m_dims.size(); i--; ) {
        if (index[i] > 0) {
            --index[i];
            return true;
        }
        index[i] = m_dims[i].extent - 1;
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

Shape Shape::broadcast(const Shape& to) const {
    int idim, ndim = static_cast<int>(to.rank());
    int idim_start = ndim - static_cast<int>(rank());

    // Can't broadcast to fewer dimensions
    if (idim_start < 0) {
        throw shape_error("could not broadcast shape");
    }

    std::vector<dim_t> new_dims(to.rank());
    size_t new_size = 1;

    for (idim = ndim - 1; idim >= idim_start; --idim) {
        auto ext = extent(idim - idim_start);
        if (ext == 1)
            new_dims[idim].stride = 0;
        else if (ext != to.extent(idim))
            throw shape_error("could not broadcast shape");
        else
            new_dims[idim].stride = stride(idim - idim_start);
        new_dims[idim].extent = to.extent(idim);
        new_size *= to.extent(idim);
    }

    // New dimensions get a zero stride
    for (idim = 0; idim < idim_start; ++idim) {
        new_dims[idim].extent = to.extent(idim);
        new_dims[idim].stride = 0;
        new_size *= to.extent(idim);
    }

    return Shape(std::move(new_dims), new_size);
}

Shape Shape::broadcast(const std::vector<Shape>& shapes) {
    // get the result shape rank
    size_t result_rank = 0;
    for (auto& shape : shapes) {
        if (shape.rank() > result_rank)
            result_rank = shape.rank();
    }

    std::vector<size_t> result_shape(result_rank);
    for (size_t i = 0; i < result_rank; i++) {
        size_t current_dim = 1;
        for (auto& shape : shapes)
            current_dim = do_broadcast(result_rank, i, current_dim, shape);
        result_shape[i] = current_dim;
    }

    return Shape(result_shape);
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << '(';
    for (auto i = 0; ; i++) {
        os << shape.extent(i);
        if (i == shape.rank()-1)
            break;
        os << ',' << ' ';
    }
    os << ')';
    return os;
}

//---------------------------------------------------------------------------

namespace detail {

ptrdiff_t shape_indexer::update(int i, ptrdiff_t& linear_idx) noexcept {
    auto dim = m_shape.extent(i);
    auto index = linear_idx % dim;
    linear_idx /= dim;
    m_offset += index * m_shape.stride(i);
    return index;
}

void shape_indexer::reset(ptrdiff_t linear_idx) noexcept {
    m_linear_idx = linear_idx;
    m_last_idx = 0;
    m_offset = 0;

    if (linear_idx > 0) {
        auto i = static_cast<int>(m_shape.rank()) - 1;
        m_last_idx = update(i, linear_idx);
        while (--i >= 0) {
            update(i, linear_idx);
        }
    }
}

void shape_indexer::increment() noexcept {
    auto linear_idx = m_linear_idx++;
    if (linear_idx < 0) {
        return;
    }

    // last dimension optimization
    auto i = static_cast<int>(m_shape.rank()) - 1;
    auto last_dim = m_shape.extent(i);
    if (++m_last_idx < last_dim) {
        m_offset += m_shape.stride(i);
        return;
    } else {
        m_last_idx = 0;
        m_offset -= m_shape.stride(i) * (last_dim - 1);
        linear_idx /= last_dim;
    }

    while (--i >= 0) {
        auto dim = m_shape.extent(i);
        auto idx = linear_idx % dim;
        if (idx != dim - 1) {
            m_offset += m_shape.stride(i);
            break;
        } else {
            m_offset -= m_shape.stride(i) * (dim - 1);
            linear_idx /= dim;
        }
    }
}

void shape_indexer::decrement() noexcept {
    auto linear_idx = m_linear_idx--;
    if (linear_idx <= 0) {
        return;
    }

    // last dimension optimization
    auto i = static_cast<int>(m_shape.rank()) - 1;
    auto last_dim = m_shape.extent(i);
    if (--m_last_idx >= 0) {
        m_offset -= m_shape.stride(i);
        return;
    } else {
        m_last_idx = last_dim - 1;
        m_offset += m_shape.stride(i) * (last_dim - 1);
        linear_idx /= last_dim;
    }

    while (--i >= 0) {
        auto dim = m_shape.extent(i);
        auto idx = linear_idx % dim;
        if (idx != 0) {
            m_offset -= m_shape.stride(i);
            break;
        } else {
            m_offset += m_shape.stride(i) * (dim - 1);
            linear_idx /= dim;
        }
    }
}

} // namespace detail
} // namespace dlf
