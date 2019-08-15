#pragma once

#include <vector>
#include <iterator>
#include <cassert>
#include <cstdlib>
#include <cstddef>
#include "utility.h"

namespace dlf {

class shape_error : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

struct SliceDim {
    int start, end, step;

    SliceDim(int start = 0, int end = std::numeric_limits<int>::max(), int step = 1) :
        start(start), end(end), step(step) {}
};

/**
 * The Shape defines the dimensions of a Tensor.
 */
class Shape final {
    struct dim_t {
        size_t extent, stride;

        bool operator==(const dim_t& rhs) const noexcept {
            return extent == rhs.extent;
        }
    };

    std::vector<dim_t> m_dims;
    size_t m_size = 0;
    size_t m_offset = 0;

    Shape(std::vector<dim_t>&& dims, size_t size, size_t offset)
        : m_dims(std::move(dims)), m_size(size), m_offset(offset) {}
    void init(const std::vector<size_t>& extents) noexcept;
    void init() noexcept;
    friend class Shaped;

public:
    Shape() = default;
    explicit Shape(const std::vector<size_t>& dims) { init(dims); };
    Shape(std::initializer_list<size_t> dims) { init(dims); }
    Shape(const Shape&) = default;
    Shape& operator=(const Shape&) = default;
    Shape(Shape&& rhs);
    Shape& operator=(Shape&& rhs);

    /**
     * Return number of dimensions in this shape.
     */
    size_t rank() const noexcept {
        return m_dims.size();
    }

    /**
     * Return shape dimensions.
     */
    std::vector<size_t> extents() const noexcept {
        std::vector<size_t> ret;
        ret.reserve(m_dims.size());
        for (auto d : m_dims)
            ret.push_back(d.extent);
        return ret;
    }

    /**
     * Returns number of elements in a given dimension.
     */
    size_t extent(size_t dim) const noexcept {
        return m_dims[dim].extent;
    }

    /**
     * Return shape strides.
     */
    std::vector<size_t> strides() const noexcept {
        std::vector<size_t> ret;
        ret.reserve(m_dims.size());
        for (auto d : m_dims)
            ret.push_back(d.stride);
        return ret;
    }

    /**
     * Returns the stride in a given dimension.
     */
    size_t stride(size_t dim) const noexcept {
        return m_dims[dim].stride;
    }

    /**
     * Returns the total number of elements defined by this shape.
     */
    size_t size() const noexcept {
        return m_size;
    }

    /**
     * Returns the partial size for the given axis range.
     */
    size_t partial_size(size_t start, size_t end) const noexcept;

    /**
     * Returns the data offset.
     */
    size_t offset() const noexcept {
        return m_offset;
    }

    /**
     * Returns true if the shape represents a contiguous addressing,
     * false otherwise.
     */
    bool is_contiguous() const noexcept;

    /**
     * Returns true if the shape is the tail of another shape.
     */
    bool is_tail(const Shape& shape) const noexcept;

    /**
     * Change dimensions of this shape. The new shape must compatible to
     * this shape.
     */
    void reshape(const std::vector<int>& new_shape);

    /**
     * Flattens the shape into a 2D matrix shape.
     */
    void flatten(int axis);

    /**
     * Remove single-dimensional entries from the shape of an array.
     *
     * @param axes Selects a subset of the single-dimensional entries
     * in the shape. If an axis is selected with shape entry greater
     * than one, an error is raised.
     */
    void squeeze(const std::vector<int> axes = {});

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_convertible<Args, int>...>::value>
    squeeze(Args... args) {
        squeeze({static_cast<int>(args)...});
    }

    /**
     * Insert single-dimensional entries to the shape.
     *
     * @param axes List of integers, indicate the dimensions to be inserted.
     */
    void unsqueeze(const std::vector<int> axes);

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_convertible<Args, int>...>::value>
    unsqueeze(Args... args) {
        unsqueeze({static_cast<int>(args)...});
    }

    /**
     * Broadcast the shape to target shape.
     */
    Shape broadcast(const Shape& to) const;

    /**
     * Create a transposed shape.
     */
    Shape transpose(const std::vector<size_t>& perm) const;
    Shape transpose() const;

    /**
     * Produces a slice of the shape along multiple axes.
     */
    Shape slice(const std::vector<int>& starts, const std::vector<int>& ends,
                const std::vector<int>& axes, const std::vector<int>& steps) const;

    /**
     * Numpy style slice.
     */
    Shape slice(const std::vector<SliceDim>& dims) const;

    /**
     * Returns the diagonal shape. The original shape must have same extent
     * on all axes.
     */
    Shape diagonal() const;

    /**
     * Returns the axis that make the give shape to be the channel of this shape.
     */
    int find_channel_axis(const Shape& base) const;

    /**
     * Return the data offset for the given index.
     */
    size_t offset(std::initializer_list<size_t> index) const noexcept;

    /**
     * Returns the data offset for the given index.
     */
    size_t offset(const std::vector<size_t>& index) const noexcept;

    /**
     * Returns the next index within this shape.
     *
     * @return true if next index is available
     */
    bool next(std::vector<size_t>& index) const noexcept;

    /**
     * Returns the previous index within this shape.
     *
     * @return true if previous index is available
     */
    bool previous(std::vector<size_t>& index) const noexcept;

    /**
     * Compare two shapes for equality.
     */
    bool operator==(const Shape& other) const noexcept {
        return m_dims == other.m_dims;
    }

    /**
     * Compare two shapes for non-equality.
     */
    bool operator!=(const Shape& other) const noexcept {
        return m_dims != other.m_dims;
    }

    friend std::ostream& operator<<(std::ostream& os, const Shape& shape);

public:
    template <typename... Shapes>
    static Shape broadcast(const Shapes&... shapes) {
        size_t result_rank = cxx::max({shapes.rank()...});
        std::vector<size_t> result_shape(result_rank);
        for (size_t i = 0; i < result_rank; i++)
            result_shape[i] = do_broadcast(result_rank, i, 1, shapes...);
        return Shape(result_shape);
    }

    static Shape broadcast(const std::vector<Shape>& shapes);

private:
    template <typename ShapeT>
    static size_t do_broadcast(size_t rank, size_t i, size_t dim, const ShapeT& shape) {
        if (i < rank - shape.rank())
            return dim; // shape will be filled with 1 at dimension i
        auto dim_i_j = shape.extent(i - rank + shape.rank());
        if (dim_i_j == 1)
            return dim;
        if (dim != dim_i_j && dim != 1)
            throw shape_error("incompatible dimensions");
        return dim_i_j;
    }

    template <typename ShapeT, typename... Shapes>
    static size_t do_broadcast(size_t rank, size_t i, size_t dim, const ShapeT& shape, const Shapes&... shapes) {
        dim = do_broadcast(rank, i, dim, shape);
        dim = do_broadcast(rank, i, dim, shapes...);
        return dim;
    }
};

/**
 * Base class for shaped objects.
 */
class Shaped {
private:
    Shape m_shape;

public:
    Shaped() = default;

    explicit Shaped(const Shape& shape, bool keep = false) :
        m_shape(shape)
    {
        if (!keep) m_shape.init();
    }

    explicit Shaped(Shape&& shape, bool keep = false) :
        m_shape(std::move(shape))
    {
        if (!keep) m_shape.init();
    }

    /**
     * Returns the shape of this shaped object.
     */
    const Shape& shape() const noexcept {
        return m_shape;
    }

    /**
     * Reshape the tensor without changing shaped data. The new shape
     * should be compatible with the original shape. At most one
     * dimension of the new shape can be -1. In this case, the
     * actual dimension value is inferred from the size of the tensor
     * and the remaining dimensions.
     *
     * @param new_shape specifies the new shape.
     * @return true if shape changed, false if new shape is not
     * compatible with current shape.
     */
    void reshape(const std::vector<int>& new_shape) {
        m_shape.reshape(new_shape);
    }

    /**
     * Flattens the tensor into a 2D matrix.
     */
    void flatten(int axis) {
        m_shape.flatten(axis);
    }

    /**
     * Remove single-dimensional entries from the shape.
     */
    void squeeze(const std::vector<int> axes = {}) {
        m_shape.squeeze(axes);
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_convertible<Args, int>...>::value>
    squeeze(Args... args) {
        m_shape.squeeze(args...);
    }

    /**
     * Insert single-dimensional entries to the shape.
     */
    void unsqueeze(const std::vector<int> axes) {
        m_shape.unsqueeze(axes);
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_convertible<Args, int>...>::value>
    unsqueeze(Args... args) {
        m_shape.unsqueeze(args...);
    }

    /**
     * Return number of dimensions in this shape.
     */
    size_t rank() const noexcept {
        return m_shape.rank();
    }

    /**
     * Returns number of elements in a given dimension.
     */
    size_t extent(size_t dim) const noexcept {
        return m_shape.extent(dim);
    }

    /**
     * Returns the stride in a given dimension.
     */
    size_t stride(size_t dim) const noexcept {
        return m_shape.stride(dim);
    }

    /**
     * Returns the total size of this tensor.
     */
    size_t size() const noexcept {
        return m_shape.size();
    }

    /**
     * Return true if this is an empty tensor.
     */
    bool empty() const noexcept {
        return size() == 0;
    }

    /**
     * Returns true if this shaped object represent a scalar.
     */
    bool is_scalar() const noexcept {
        return rank() == 1 && extent(0) == 1;
    }

    /**
     * Returns true if this shaped object represent a 1-dimensional vector.
     */
    bool is_vector() const noexcept {
        return rank() == 1;
    }

    /**
     * Returns true if this shaped object represents a 2-dimensional matrix.
     */
    bool is_matrix() const noexcept {
        return rank() == 2;
    }

    /**
     * Returns true if this shaped object is a square matrix.
     */
    bool is_square() const noexcept {
        return is_matrix() && extent(0) == extent(1);
    }
};

//---------------------------------------------------------------------------
// Shaped iterator

namespace detail {
class shape_indexer {
private:
    ptrdiff_t m_linear_idx;
    ptrdiff_t m_last_idx;
    size_t    m_offset;

protected:
    Shape m_shape;

    shape_indexer(const Shape& shape, ptrdiff_t start)
        : m_shape(shape) { reset(start); }
    shape_indexer(Shape&& shape, ptrdiff_t start)
        : m_shape(std::move(shape)) { reset(start);}

    void reset(ptrdiff_t linear_idx) noexcept;
    void increment() noexcept;
    void decrement() noexcept;
    void increment(ptrdiff_t n) noexcept { reset(m_linear_idx + n); }
    void decrement(ptrdiff_t n) noexcept { reset(m_linear_idx - n); }

public:
    ptrdiff_t index() const noexcept { return m_linear_idx; }
    size_t offset() const noexcept { return m_offset; }

    bool operator==(const shape_indexer& rhs) const noexcept {
        return m_linear_idx == rhs.m_linear_idx;
    }
    bool operator!=(const shape_indexer& rhs) const noexcept {
        return m_linear_idx != rhs.m_linear_idx;
    }
    bool operator<(const shape_indexer& rhs) const noexcept {
        return m_linear_idx < rhs.m_linear_idx;
    }
    bool operator>(const shape_indexer& rhs) const noexcept {
        return m_linear_idx > rhs.m_linear_idx;
    }
    bool operator<=(const shape_indexer& rhs) const noexcept {
        return m_linear_idx <= rhs.m_linear_idx;
    }
    bool operator>=(const shape_indexer& rhs) const noexcept {
        return m_linear_idx >= rhs.m_linear_idx;
    }

private:
    ptrdiff_t update(int i, ptrdiff_t& linear_idx) noexcept;
};
}

template <typename T>
class shaped_iterator : public detail::shape_indexer {
    T* m_data;

public:
    using value_type = T;
    using difference_type = ptrdiff_t;
    using reference = value_type&;
    using pointer = value_type*;
    using iterator_category = std::random_access_iterator_tag;

    shaped_iterator(const Shape& shape, T* data, difference_type start)
        : shape_indexer(shape, start), m_data(data) {}
    shaped_iterator(Shape&& shape, T* data, difference_type start)
        : shape_indexer(std::move(shape), start), m_data(data) {}

    shaped_iterator& operator++() noexcept { increment(); return *this; }
    shaped_iterator& operator--() noexcept { decrement(); return *this; }
    shaped_iterator  operator++(int) noexcept { auto t = *this; ++(*this); return t; }
    shaped_iterator  operator--(int) noexcept { auto t = *this; --(*this); return t; }
    shaped_iterator& operator+=(difference_type n) noexcept { increment(n); return *this; }
    shaped_iterator& operator-=(difference_type n) noexcept { decrement(n); return *this; }

    shaped_iterator operator+(difference_type n) const noexcept
        { return shaped_iterator(m_shape, m_data, index() + n); }
    shaped_iterator operator-(difference_type n) const noexcept
        { return shaped_iterator(m_shape, m_data, index() - n); }
    difference_type operator-(const shaped_iterator& rhs) const noexcept
        { return index() - rhs.index(); }

    reference operator*() const noexcept {
        assert(index() >= 0 && index() < m_shape.size());
        return m_data[offset()];
    }

    pointer operator->() const noexcept {
        assert(index() >= 0 && index() < m_shape.size());
        return &m_data[offset()];
    }
};

template <typename T>
class const_shaped_iterator : public detail::shape_indexer {
    const T* m_data;

public:
    using value_type = T;
    using difference_type = ptrdiff_t;
    using reference = const value_type&;
    using pointer = const value_type*;
    using iterator_category = std::random_access_iterator_tag;

    const_shaped_iterator(const Shape& shape, const T* data, difference_type start)
        : shape_indexer(shape, start), m_data(data) {}
    const_shaped_iterator(Shape&& shape, T* data, difference_type start)
        : shape_indexer(std::move(shape), start), m_data(data) {}

    const_shaped_iterator& operator++() noexcept { increment(); return *this; }
    const_shaped_iterator& operator--() noexcept { decrement(); return *this; }
    const_shaped_iterator  operator++(int) noexcept { auto t = *this; ++(*this); return t; }
    const_shaped_iterator  operator--(int) noexcept { auto t = *this; --(*this); return t; }
    const_shaped_iterator& operator+=(difference_type n) noexcept { increment(n); return *this; }
    const_shaped_iterator& operator-=(difference_type n) noexcept { decrement(n); return *this; }

    const_shaped_iterator operator+(difference_type n) const noexcept
        { return const_shaped_iterator(m_shape, m_data, index() + n); }
    const_shaped_iterator operator-(difference_type n) const noexcept
        { return const_shaped_iterator(m_shape, m_data, index() - n); }
    difference_type operator-(const const_shaped_iterator& rhs) const noexcept
        { return index() - rhs.index(); }

    reference operator*() const noexcept {
        assert(index() >= 0 && index() < m_shape.size());
        return m_data[offset()];
    }

    pointer operator->() const noexcept {
        assert(index() >= 0 && index() < m_shape.size());
        return &m_data[offset()];
    }
};

} // namespace dlf
