#pragma once

namespace dlf {

class shape_error : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

struct Range {
    int start = 0;
    int end = std::numeric_limits<int>::max();
    int step = 1;

    Range() = default;
    Range(int start) : start(start), end(start), step(1) {}
    Range(int start, int end, int step = 1) :
        start(start), end(end), step(step) {}

    Range normalize(int max_item) const noexcept {
        int start = this->start;
        if (start < 0)
            start += max_item;
        if (step < 0)
            start = cxx::clamp(start, 0, max_item-1);
        else
            start = cxx::clamp(start, 0, max_item);

        int end = this->end;
        if (end < 0)
            end += max_item;
        if (step < 0)
            end = cxx::clamp(end, -1, max_item);
        else
            end = cxx::clamp(end, 0, max_item);

        return Range(start, end, step);
    }

    int size() const noexcept {
        int len = (end - start - (step<0 ? -1 : 1)) / step + 1;
        if (len == 0)
            len = 1;
        return len;
    }
};

/**
 * The Shape defines the dimensions of a Spatial.
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

    void init(const std::vector<size_t>& extents, size_t offset = 0) noexcept;
    void init() noexcept;

    template <typename Derived> friend class Spatial;

public:
    Shape() : m_size(1) {} // create a scalar shape

    explicit Shape(const std::vector<size_t>& dims) {
        init(dims);
    }

    Shape(std::initializer_list<size_t> dims) {
        init(dims);
    }

    template <typename... Args, typename = std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value>>
    explicit Shape(Args... args) {
        init(std::vector<size_t>{static_cast<size_t>(args)...});
    }

    static Shape as_strided(const std::vector<size_t>& extents, const std::vector<size_t>& strides, size_t offset);

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
    size_t extent(int axis) const noexcept {
        if (axis < 0) axis += rank();
        assert(axis >= 0 && axis < rank());
        return m_dims[axis].extent;
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
    size_t stride(int axis) const noexcept {
        if (axis < 0) axis += rank();
        assert(axis >= 0 && axis < rank());
        return m_dims[axis].stride;
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
     * Returns a shape with new dimensions. The new shape must compatible to
     * this shape.
     */
    Shape reshape(const std::vector<int>& new_shape) const;
    Shape reshape(Shape new_shape) const;

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, Shape>
    reshape(Args... args) const {
        return reshape(std::vector<int>{static_cast<int>(args)...});
    }

    /**
     * Flattens the shape into a 2D matrix shape.
     */
    Shape flatten(int axis) const;

    /**
     * Remove single-dimensional entries from the shape of an array.
     *
     * @param axes Selects a subset of the single-dimensional entries
     * in the shape. If an axis is selected with shape entry greater
     * than one, an error is raised.
     */
    Shape squeeze(const std::vector<int>& axes) const;
    Shape squeeze(int axis) const;

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, Shape>
    squeeze(Args... args) const {
        return squeeze(std::vector<int>{static_cast<int>(args)...});
    }

    /**
     * Insert single-dimensional entries to the shape.
     *
     * @param axes List of integers, indicate the dimensions to be inserted.
     */
    Shape unsqueeze(const std::vector<int>& axes) const;
    Shape unsqueeze(int axis) const;

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, Shape>
    unsqueeze(Args... args) const {
        return unsqueeze(std::vector<int>{static_cast<int>(args)...});
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

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, Shape>
    transpose(Args... args) const {
        return transpose({static_cast<size_t>(args)...});
    }

    /**
     * Produces a slice of the shape along multiple axes.
     */
    Shape slice(const std::vector<int>& starts, const std::vector<int>& ends,
                const std::vector<int>& axes, const std::vector<int>& steps) const;

    /**
     * Numpy style slice.
     */
    Shape slice(const std::vector<Range>& range) const;
    Shape slice(const char* spec) const;

    /**
     * Returns the diagonal shape.
     *
     * This the shape is 2-D, returns the diagonal of the shape with the given
     * offset. If the shape has more then two dimensions, then the axes
     * specified by axis1 and axis2 are used to determine the 2-D sub matrix
     * whose diagonal is returned. The shape of the resulting tensor can be
     * determined by removing axis1 and axis2 and appending an index to the
     * right equal to the size of the resulting diagonals.
     *
     * @param offset Offsets of the diagonal from the main diagonal. Can be
     *        positive or negative. Defaults to main diagonal (0).
     * @param axis1 Axis to be used as the first axis of the 2-D sub-matrix
     *        from which the diagonals should be taken. Defaults to last of
     *        second axis (-2).
     * @param axis2 Axis to be used as the second axis of the 2-D sub-matrix
     *        from which the diagonals should be taken. Defaults to the last
     *        axis (-1).
     */
    Shape diagonal(int offset = 0, int axis1 = -2, int axis2 = -1) const;

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
     * Returns the data offset for the given linear index.
     */
    size_t linear_offset(size_t index) const noexcept;

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
 * Base class for spatial objects.
 */
template <typename Derived>
class Spatial {
private:
    Shape m_shape;

public:
    Spatial() { m_shape.m_size = 0; } // initialize to empty shape

    explicit Spatial(const Shape& shape, bool keep = false)
        : m_shape(shape) { if (!keep) m_shape.init(); }
    explicit Spatial(Shape&& shape, bool keep = false)
        : m_shape(std::move(shape)) { if (!keep) m_shape.init(); }

    /**
     * Returns the shape of this shaped object.
     */
    const Shape& shape() const noexcept {
        return m_shape;
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
    size_t extent(int axis) const noexcept {
        return m_shape.extent(axis);
    }

    /**
     * Returns the stride in a given dimension.
     */
    size_t stride(int axis) const noexcept {
        return m_shape.stride(axis);
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
        return rank() == 0;
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

    /**
     * Returns true if last two dimension of then tensor is equal.
     */
    bool is_inner_square() const noexcept {
        return rank() >= 2 && extent(rank()-1) == extent(rank()-2);
    }

protected:
    /**
     * Reshape the tensor without changing shaped data. The new shape
     * should be compatible with the original shape.
     *
     * @param new_shape specifies the new shape.
     */
    void set_shape(Shape new_shape) {
        m_shape = std::move(new_shape);
        m_shape.init();
    }

    /**
     * Reshape the tensor without changing shaped data. The new shape
     * should be compatible with the original shape. At most one
     * dimension of the new shape can be -1. In this case, the
     * actual dimension value is inferred from the size of the tensor
     * and the remaining dimensions.
     *
     * @param new_shape specifies the new shape.
     */
    template <typename... Args>
    void reshape(Args&&... args) {
        set_shape(m_shape.reshape(std::forward<Args>(args)...));
    }

    /**
     * Flattens the tensor into a 2D matrix.
     */
    void flatten(int axis) {
        set_shape(m_shape.flatten(axis));
    }

    /**
     * Flattens the tensor into a 1D vector.
     */
    void flatten() {
        reshape(-1);
    }

    /**
     * Remove single-dimensional entries from the shape.
     */
    template <typename... Args>
    void squeeze(Args&&... args) {
        set_shape(m_shape.squeeze(std::forward<Args>(args)...));
    }

    /**
     * Insert single-dimensional entries to the shape.
     */
    template <typename... Args>
    void unsqueeze(Args&&... args) {
        set_shape(m_shape.unsqueeze(std::forward<Args>(args)...));
    }

public:
    auto broadcast(const Shape& to) const {
        return self().view(m_shape.broadcast(to));
    }

    template <typename... Args>
    auto transpose(Args&&... args) const {
        return self().view(m_shape.transpose(std::forward<Args>(args)...));
    }

    auto slice(const std::vector<int>& starts, const std::vector<int>& ends,
               const std::vector<int>& axes, const std::vector<int>& steps) const {
        return self().view(m_shape.slice(starts, ends, axes, steps));
    }

    auto slice(const std::vector<Range>& range) const {
        return self().view(m_shape.slice(range));
    }

    auto operator[](const char* spec) const {
        return self().view(m_shape.slice(spec));
    }

    auto operator[](const std::string& spec) const {
        return operator[](spec.c_str());
    }

    auto operator[](const int index) const {
        return row(index);
    }

    auto row(int index) const {
        return self().view(m_shape.slice({{index}}).squeeze(0));
    }

    auto column(int index) const {
        return self().view(m_shape.slice({{}, {index}}).squeeze(1));
    }

    auto diagonal(int offset = 0, int axis1 = -2, int axis2 = -1) const {
        return self().view(m_shape.diagonal(offset, axis1, axis2));
    }

private:
    inline const Derived& self() const {
        return *static_cast<const Derived*>(this);
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

    bool operator==(const shape_indexer& rhs) const noexcept
        { return m_linear_idx == rhs.m_linear_idx; }
    bool operator!=(const shape_indexer& rhs) const noexcept
        { return m_linear_idx != rhs.m_linear_idx; }
    bool operator<(const shape_indexer& rhs) const noexcept
        { return m_linear_idx < rhs.m_linear_idx; }
    bool operator>(const shape_indexer& rhs) const noexcept
        { return m_linear_idx > rhs.m_linear_idx; }
    bool operator<=(const shape_indexer& rhs) const noexcept
        { return m_linear_idx <= rhs.m_linear_idx; }
    bool operator>=(const shape_indexer& rhs) const noexcept
        { return m_linear_idx >= rhs.m_linear_idx; }

private:
    ptrdiff_t update(int i, ptrdiff_t& linear_idx) noexcept;
};
} // namespace detail

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

template <typename T>
class strided_iterator {
    T* m_data;
    size_t m_stride;

public:
    using value_type = T;
    using difference_type = ptrdiff_t;
    using reference = value_type&;
    using pointer = value_type*;
    using iterator_category = std::random_access_iterator_tag;

    strided_iterator(T* data, size_t stride, difference_type start)
        : m_data(data + start*stride), m_stride(stride) {}

    strided_iterator& operator++() noexcept { m_data += m_stride; return *this; }
    strided_iterator& operator--() noexcept { m_data -= m_stride; return *this; }
    strided_iterator  operator++(int) noexcept { auto t = *this; ++(*this); return t; }
    strided_iterator  operator--(int) noexcept { auto t = *this; --(*this); return t; }
    strided_iterator& operator+=(difference_type n) noexcept { m_data += n*m_stride; return *this; }
    strided_iterator& operator-=(difference_type n) noexcept { m_data -= n*m_stride; return *this; }

    strided_iterator operator+(difference_type n) const noexcept
        { return strided_iterator(m_data, m_stride, n); }
    strided_iterator operator-(difference_type n) const noexcept
        { return strided_iterator(m_data, m_stride, -n); }
    difference_type operator-(const strided_iterator& rhs) const noexcept
        { return (m_data - rhs.m_data) / m_stride; }

    reference operator*() const noexcept { return *m_data; }
    pointer operator->() const noexcept { return m_data; }
    reference operator[](int n) const noexcept { return m_data[n*m_stride]; }

    bool operator==(const strided_iterator& rhs) const noexcept
        { return m_data == rhs.m_data; }
    bool operator!=(const strided_iterator& rhs) const noexcept
        { return m_data != rhs.m_data; }
    bool operator<(const strided_iterator& rhs) const noexcept
        { return m_data < rhs.m_data; }
    bool operator<=(const strided_iterator& rhs) const noexcept
        { return m_data <= rhs.m_data; }
    bool operator>(const strided_iterator& rhs) const noexcept
        { return m_data > rhs.m_data; }
    bool operator>=(const strided_iterator& rhs) const noexcept
        { return m_data >= rhs.m_data; }
};

template <typename T>
class const_strided_iterator {
    const T* m_data;
    size_t m_stride;

public:
    using value_type = T;
    using difference_type = ptrdiff_t;
    using reference = const value_type&;
    using pointer = const value_type*;
    using iterator_category = std::random_access_iterator_tag;

    const_strided_iterator(const T* data, size_t stride, difference_type start)
        : m_data(data + start*stride), m_stride(stride) {}

    const_strided_iterator& operator++() noexcept { m_data += m_stride; return *this; }
    const_strided_iterator& operator--() noexcept { m_data -= m_stride; return *this; }
    const_strided_iterator  operator++(int) noexcept { auto t = *this; ++(*this); return t; }
    const_strided_iterator  operator--(int) noexcept { auto t = *this; --(*this); return t; }
    const_strided_iterator& operator+=(difference_type n) noexcept { m_data += n*m_stride; return *this; }
    const_strided_iterator& operator-=(difference_type n) noexcept { m_data -= n*m_stride; return *this; }

    const_strided_iterator operator+(difference_type n) const noexcept
        { return strided_iterator(m_data, m_stride, n); }
    const_strided_iterator operator-(difference_type n) const noexcept
        { return strided_iterator(m_data, m_stride, -n); }
    difference_type operator-(const const_strided_iterator& rhs) const noexcept
        { return (m_data - rhs.m_data) / m_stride; }

    reference operator*() const noexcept { return *m_data; }
    pointer operator->() const noexcept { return m_data; }
    reference operator[](int n) const noexcept { return m_data[n*m_stride]; }

    bool operator==(const const_strided_iterator& rhs) const noexcept
        { return m_data == rhs.m_data; }
    bool operator!=(const const_strided_iterator& rhs) const noexcept
        { return m_data != rhs.m_data; }
    bool operator<(const const_strided_iterator& rhs) const noexcept
        { return m_data < rhs.m_data; }
    bool operator<=(const const_strided_iterator& rhs) const noexcept
        { return m_data <= rhs.m_data; }
    bool operator>(const const_strided_iterator& rhs) const noexcept
        { return m_data > rhs.m_data; }
    bool operator>=(const const_strided_iterator& rhs) const noexcept
        { return m_data >= rhs.m_data; }
};

//---------------------------------------------------------------------------

namespace detail {
void norm_axis(const int rank, int& axis);
void norm_axes(const int rank, int& axis1, int& axis2, bool allow_duplicates = false);
void norm_axes(const int rank, std::vector<int>& axes, bool allow_duplicates = false);
} // namespace detail

} // namespace dlf
