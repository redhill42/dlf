#include <unordered_set>
#include <cstring>
#include <numeric>
#include "tensor/shape.h"
#include "utility.h"

namespace dlf {

Shape::Shape(Shape&& rhs)
    : m_dims(std::move(rhs.m_dims)),
      m_size(cxx::exchange(rhs.m_size, 0)),
      m_offset(cxx::exchange(rhs.m_offset, 0))
{
}

Shape& Shape::operator=(Shape&& rhs) {
    m_dims = std::move(rhs.m_dims);
    m_size = cxx::exchange(rhs.m_size, 0);
    m_offset = cxx::exchange(rhs.m_offset, 0);
    return *this;
}

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
    m_offset = 0;
}

void Shape::init() noexcept {
    size_t size = 1;
    for (size_t i = m_dims.size(); i-- != 0; ) {
        m_dims[i].stride = size;
        size *= m_dims[i].extent;
    }
    m_offset = 0;
}

Shape Shape::as_strided(const std::vector<size_t>& extents, const std::vector<size_t>& strides) {
    assert(extents.size() == strides.size());
    std::vector<dim_t> dims(extents.size());

    for (size_t i = 0; i < extents.size(); ++i) {
        assert(extents[i] != 0);
        dims[i].extent = extents[i];
        dims[i].stride = strides[i];
    }

    auto size = std::accumulate(extents.begin(), extents.end(), 1, std::multiplies<>());
    return Shape(std::move(dims), size, 0);
}

size_t Shape::partial_size(size_t start, size_t end) const noexcept {
    assert(start >= 0 && start <= rank());
    assert(end >= start && end <= rank());
    size_t res = 1;
    for (size_t i = start; i < end; i++)
        res *= extent(i);
    return res;
}

bool Shape::is_contiguous() const noexcept {
    size_t size = 1;
    for (int i = rank(); --i >= 0;) {
        if (stride(i) == 0 && extent(i) == 1)
            continue;
        if (stride(i) != size)
            return false;
        size *= extent(i);
    }
    return true;
}

bool Shape::is_tail(const dlf::Shape& shape) const noexcept {
    // scalar is always the tail of another shape
    if (size() == 1)
        return rank() <= shape.rank();

    int idim, ndim = static_cast<int>(shape.rank());
    int idim_start = ndim - static_cast<int>(rank());

    // Can't be tail of fewer dimensions
    if (idim_start < 0) {
        return false;
    }

    for (idim = ndim - 1; idim >= idim_start; --idim) {
        if (extent(idim - idim_start) != shape.extent(idim))
            return false;
        if (stride(idim - idim_start) != shape.stride(idim))
            return false;
    }
    return true;
}

size_t Shape::offset(std::initializer_list<size_t> index) const noexcept {
    assert(index.size() == m_dims.size());
    size_t offset = m_offset, i = 0;
    for (auto a : index) {
        offset += a * m_dims[i].stride;
        ++i;
    }
    return offset;
}

size_t Shape::offset(const std::vector<size_t>& index) const noexcept {
    assert(index.size() == m_dims.size());
    size_t offset = m_offset, i = 0;
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

int Shape::find_channel_axis(const Shape& base) const {
    int axis = -1;
    if (base.rank() <= rank()) {
        for (int i = base.rank(); --i >= 0; ) {
            if (base.extent(i) != 1) {
                if (axis != -1)
                    return -1;
                axis = i + rank() - base.rank();
                if (base.extent(i) != extent(axis))
                    return -1;
            }
        }
    }
    return axis;
}

Shape Shape::reshape(const std::vector<int>& new_shape) const {
    std::vector<size_t> dims(new_shape.size());
    size_t new_size = 1;
    int pending = -1;

    if (new_shape.size() == 0) {
        new_size = 0;
    } else {
        for (int i = 0; i < new_shape.size(); i++) {
            if (new_shape[i] < 0) {
                if (new_shape[i] != -1 || pending != -1)
                    throw shape_error("reshape: invalid shape");
                pending = i;
            } else {
                if (new_shape[i] == 0) {
                    if (i >= rank())
                        throw shape_error("reshape: incompatible shape");
                    dims[i] = this->extent(i);
                } else {
                    dims[i] = new_shape[i];
                }
                new_size *= dims[i];
            }
        }
    }

    if (pending != -1) {
        dims[pending] = size() / new_size;
        new_size *= dims[pending];
    }

    if (size() != new_size)
        throw shape_error("reshape: incompatible shape");
    return Shape(std::move(dims));
}

Shape Shape::flatten(int axis) const {
    if (axis < 0) axis += rank();
    if (axis < 0 || axis > rank())
        throw shape_error("flatten: invalid axis");

    size_t rows = partial_size(0, axis);
    size_t cols = partial_size(axis, rank());
    return Shape(rows, cols);
}

Shape Shape::squeeze(const std::vector<int> axes) const {
    std::unordered_set<int> norm_axes;
    for (auto a : axes) {
        if (a < 0) a += rank();
        if (a < 0 || a >= rank())
            throw shape_error("squeeze: Invalid axis");
        norm_axes.insert(a); // duplicate is ok
    }

    std::vector<dim_t> new_dims;
    for (int i = 0; i < rank(); i++) {
        if (norm_axes.find(i) != norm_axes.end()) {
            if (extent(i) != 1)
                throw shape_error("squeeze: cannot select an axis to squeeze out which has size not equal to one");
            continue;
        } else if (norm_axes.empty() && extent(i) == 1) {
            continue;
        } else {
            new_dims.push_back(m_dims[i]);
        }
    }

    return Shape(std::move(new_dims), size(), offset());
}

Shape Shape::unsqueeze(const std::vector<int> axes) const {
    const auto new_rank = rank() + axes.size();
    std::unordered_set<int> norm_axes;

    for (auto a : axes) {
        if (a < 0) a += new_rank;
        if (a < 0 || a >= new_rank)
            throw shape_error("unsqueeze: Invalid axis");
        if (norm_axes.find(a) != norm_axes.end())
            throw shape_error("unsqueeze: Duplicate axis value");
        norm_axes.insert(a);
    }

    std::vector<dim_t> new_dims;
    for (size_t i = 0, j = 0; i < new_rank; i++) {
        if (norm_axes.find(i) != norm_axes.end()) {
            new_dims.push_back({1, 0});
        } else {
            new_dims.push_back(m_dims[j++]);
        }
    }

    return Shape(std::move(new_dims), size(), offset());
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

    return Shape(std::move(new_dims), new_size, offset());
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

static bool validate_perm(size_t rank, const std::vector<size_t>& perm) {
    if (perm.size() != rank)
        return false;
    std::unordered_set<size_t> unique_index;
    for (auto index : perm) {
        if (!(0 <= index && index < rank))
            return false;
        if (unique_index.find(index) != unique_index.end())
            return false;
        unique_index.insert(index);
    }
    return true;
}

Shape Shape::transpose(const std::vector<size_t>& perm) const {
    if (!validate_perm(rank(), perm)) {
        throw shape_error("transpose: invalid permutation");
    }
    std::vector<dim_t> dims(rank());
    for (size_t i = 0; i < rank(); i++) {
        dims[i] = m_dims[perm[i]];
    }
    return Shape(std::move(dims), size(), offset());
}

Shape Shape::transpose() const {
    if (rank() == 1) {
        return Shape({
            {extent(0), stride(0)},
            {1, 0}
        }, size(), offset());
    }

    std::vector<size_t> perm(rank());
    std::iota(perm.begin(), perm.end(), 0);
    std::reverse(perm.begin(), perm.end());
    return transpose(perm);
}

Shape Shape::slice(
    const std::vector<int>& starts, const std::vector<int>& ends,
    const std::vector<int>& axes_opt, const std::vector<int>& steps_opt) const
{
    if (starts.size() != ends.size() || starts.size() > rank())
        throw shape_error("slice: incorrect value for starts and ends");

    std::vector<int> axes;
    if (axes_opt.empty()) {
        axes.resize(starts.size());
        std::iota(axes.begin(), axes.end(), 0);
    } else {
        if (axes_opt.size() != starts.size())
            throw shape_error("slice: axes has incorrect length");
        axes = axes_opt;
    }

    std::vector<int> steps;
    if (steps_opt.empty()) {
        steps.resize(starts.size());
        std::fill(steps.begin(), steps.end(), 1);
    } else {
        if (steps_opt.size() != starts.size())
            throw shape_error("slice: steps has incorrect length");
        if (std::any_of(steps_opt.begin(), steps_opt.end(), [](auto x){ return x == 0; }))
            throw shape_error("slice: step cannot be 0");
        steps = steps_opt;
    }

    std::vector<dim_t> dims = m_dims;
    std::vector<size_t> start_index(rank(), 0);
    std::unordered_set<int> unique_axes;

    for (int i = 0; i < axes.size(); ++i) {
        auto axis = axes[i];
        if (axis < 0) axis += rank();
        if (axis < 0 || axis >= rank())
            throw shape_error("slice: axes has invalid value");
        if (unique_axes.find(axis) != unique_axes.end())
            throw shape_error("slice: axes has duplicates");
        unique_axes.insert(axis);

        auto input_dim = static_cast<int>(extent(axis));
        auto step = steps[i];

        int start = starts[i];
        if (start < 0)
            start += input_dim;
        if (step < 0)
            start = cxx::clamp(start, 0, input_dim - 1);
        else
            start = cxx::clamp(start, 0, input_dim);

        int end = ends[i];
        if (end < 0)
            end += input_dim;
        if (step < 0)
            end = cxx::clamp(end, -1, input_dim);
        else
            end = cxx::clamp(end, 0, input_dim);

        // find output dim value for this axis
        auto temp = (end - start - (step<0 ? -1 : 1)) / step + 1;
        if (temp <= 0)
            throw shape_error("slice: incorrect start and end value");
        dims[axis].extent = temp;
        dims[axis].stride *= step;
        start_index[axis] = start;
    }

    size_t size = 1;
    for (int i = 0; i < rank(); i++)
        size *= dims[i].extent;
    return Shape(std::move(dims), size, offset(start_index));
}

Shape Shape::slice(const std::vector<SliceDim>& dims) const {
    std::vector<int> axes(dims.size());
    std::vector<int> starts(dims.size());
    std::vector<int> ends(dims.size());
    std::vector<int> steps(dims.size());

    for (int i = 0; i < dims.size(); i++) {
        axes[i] = i;
        starts[i] = dims[i].start;
        ends[i] = dims[i].end;
        steps[i] = dims[i].step;
    }

    return slice(starts, ends, axes, steps);
}

Shape Shape::diagonal(int offset, int axis1, int axis2) const {
    if (rank() < 2)
        throw shape_error("diagonal: requires at least rank-2");
    if (axis1 < 0) axis1 += rank();
    if (axis1 < 0 || axis1 >= rank())
        throw shape_error("diagonal: invalid axis1 value");
    if (axis2 < 0) axis2 += rank();
    if (axis2 < 0 || axis2 >= rank())
        throw shape_error("diagonal: invalid axis2 value");
    if (axis1 == axis2)
        throw shape_error("diagonal: the axis1 and axis2 should not have the same value");

    int k = std::min(extent(axis1), extent(axis2));
    if ((k -= std::abs(offset)) <= 0) {
        throw shape_error("diagonal: invalid offset value");
    }

    auto s = stride(axis1) + stride(axis2);
    size_t shape_offset;
    if (offset >= 0) {
        shape_offset = this->offset() + offset * stride(axis2);
    } else {
        shape_offset = this->offset() - offset * stride(axis1);
    }

    // transpose axis1 and axis2 to end of shape
    std::vector<size_t> perm;
    size_t shape_size = k;
    for (size_t i = 0; i < rank(); i++) {
        if (i != axis1 && i != axis2) {
            perm.push_back(i);
            shape_size *= extent(i);
        }
    }
    perm.push_back(axis1);
    perm.push_back(axis2);

    auto dims = transpose(perm).m_dims;
    dims.erase(dims.end()-2, dims.end());
    dims.push_back({static_cast<size_t>(k), s});
    return Shape(std::move(dims), shape_size, shape_offset);
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    if (shape.rank() == 0) {
        os << "<<>>";
        return os;
    }

    os << "<<";
    for (auto i = 0; ; i++) {
        os << shape.extent(i);
        if (i == shape.rank()-1)
            break;
        os << ',';
    }
    os << ">>";
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
    m_offset = m_shape.offset();

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
