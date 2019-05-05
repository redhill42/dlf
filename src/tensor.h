#ifndef KNERON_TENSOR_H
#define KNERON_TENSOR_H

#include <vector>
#include <iostream>

namespace kneron::model {

/**
 * The Shape defines the dimensions of a Tensor.
 */
class Shape
{
    std::vector<size_t> m_dims;

public:
    Shape() = default;
    explicit Shape(std::vector<size_t> init) : m_dims(std::move(init)) {}
    Shape(std::initializer_list<size_t> init) : m_dims(init) {}

    /**
     * Return number of dimensions in this shape.
     */
    size_t rank() const noexcept {
        return m_dims.size();
    }

    /**
     * Returns number of elements in a given dimension.
     */
    size_t operator[](size_t dim) const noexcept {
        return m_dims[dim];
    }

    bool operator==(const Shape& other) const {
        return m_dims == other.m_dims;
    }

    bool operator!=(const Shape& other) const {
        return m_dims != other.m_dims;
    }

    /**
     * Shrink one level of dimensions.
     */
    Shape shrink() const {
        return Shape(std::vector<size_t>(m_dims.begin()+1, m_dims.end()));
    }

    /**
     * Returns the data size defined by this shape.
     */
    size_t size() const noexcept;

    /**
     * Return the data offset for the given index.
     */
    size_t offset(const std::initializer_list<size_t>& index) const noexcept;

    /**
     * Returns the next index within this shape.
     *
     * @return true if next index is available
     */
    bool next(std::vector<size_t>& index) const noexcept;
};

template <typename InputIterator, typename T>
using RequireInputIterator =
    std::enable_if_t<
        std::is_convertible_v<
            typename std::iterator_traits<InputIterator>::iterator_category,
            std::input_iterator_tag> &&
        std::is_constructible_v<
            T, typename std::iterator_traits<InputIterator>::reference>,
        InputIterator>;

/**
 * Tensor is a geometric object that maps in a multi-linear manner geometric
 * vectors, scalars, and other tensors to a resulting tensor.
 *
 * @tparam T the data type of the tensor.
 */
template <typename T>
class Tensor {
    Shape m_shape;
    size_t m_size = 0;
    T* m_data = nullptr;
    std::unique_ptr<T[]> m_alloc_data;

public: // Container View
    using value_type                = T;
    using reference                 = value_type&;
    using const_reference           = const value_type&;
    using iterator                  = value_type*;
    using const_iterator            = const value_type*;
    using pointer                   = value_type*;
    using const_pointer             = const value_type*;
    using size_type                 = size_t;
    using difference_type           = ptrdiff_t;
    using reverse_iterator          = std::reverse_iterator<iterator>;
    using const_reverse_iterator    = std::reverse_iterator<const_iterator>;

    iterator begin() noexcept { return iterator(data()); }
    const_iterator begin() const noexcept { return const_iterator(data()); }
    iterator end() noexcept { return iterator(data() + size()); }
    const_iterator end() const noexcept { return const_iterator(data() + size()); }

    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }
    const_reverse_iterator crbegin() const noexcept { return rbegin(); }
    const_reverse_iterator crend() const noexcept { return rend(); }

private:
    /**
     * Construct a tensor with given dimension and wrapped data. This constructor
     * can only be called from wrap() function.
     *
     * @param shape the tensor dimensions
     * @param data the tensor data
     */
    Tensor(Shape shape, T* data);

public:
    /**
     * Construct a 0-dimensional tensor.
     */
    Tensor() = default;

    /**
     * Construct a tensor with given dimensions.
     *
     * @param shape the tensor shape
     */
    explicit Tensor(Shape shape);

    /**
     * Construct a tensor with input iterator denoted by [begin,end).
     *
     * @param shape the tensor dimensions
     * @param begin the start of input iterator
     * @param end the end of input iterator
     */
    template <typename It>
    Tensor(Shape shape, It begin, RequireInputIterator<It,T> end);

    /**
     * Construct a tensor with an initializer list.
     *
     * @param shape the tensor dimensions
     * @param init the initializer list
     */
    Tensor(Shape shape, std::initializer_list<T> init)
        : Tensor(std::move(shape), init.begin(), init.end())
    {}

    /**
     * Construct a tensor with given dimension and preallocated data. The ownership
     * of the data is transferred to this tensor and the data should not be used by
     * caller again. It's the caller's responsibility to allocate enough memory
     * space to store the tensor data, and encapsulate the data into a unique_ptr.
     * The memory space allocated by caller will be freed when this tensor is no
     * longer used.
     *
     * @param shape the tensor dimensions.
     * @param data the preallocated tensor data.
     */
    Tensor(Shape shape, std::unique_ptr<T[]> data);

    /**
     * Wraps a raw data as a tensor, given the dimension of tensor. This constructor
     * is convenient to wrap an existing tensor data to perform tensor computation.
     * This tensor doesn't own the data. It must be sure that the data is valid during
     * the lifecycle of this tensor, otherwise the behavior is unspecified.
     *
     * @param shape the tensor dimension
     * @param data the wrapped tensor data.
     */
    static Tensor wrap(Shape shape, T* data);

    /**
     * Build the tensor with given generator function. The generator function
     * accepts an index and a sequence number which can aid to generate the
     * tensor data.
     *
     * @param shape the tensor dimension
     * @param f the generator function
     * @return the generated tensor
     */
    template <typename F>
    static Tensor build(Shape shape, F f);

    // Copy and move constructors/assignments.
    Tensor(const Tensor& t);
    Tensor& operator=(const Tensor& t);
    Tensor(Tensor&& t) noexcept;
    Tensor& operator=(Tensor&& t) noexcept;

    /**
     * Returns the shape of this tensor.
     */
    const Shape& shape() const noexcept {
        return m_shape;
    }

    /**
     * Return true if this is an empty tensor.
     */
    bool empty() const noexcept {
        return m_size == 0;
    }

    /**
     * Returns true if this tensor represent a 1-dimensional vector.
     */
    bool is_vector() const noexcept {
        return m_shape.rank() == 1;
    }

    /**
     * Returns true if this tensor represents a 2-dimensional matrix.
     */
    bool is_matrix() const noexcept {
        return m_shape.rank() == 2;
    }

    /**
     * Returns the total size of this tensor.
     */
    size_t size() const noexcept {
        return m_size;
    }

    /**
     * Returns the maximal size.
     */
    size_t max_size() const noexcept {
        return m_size;
    }

    /**
     * Returns the raw data elements.
     */
    T* data() noexcept {
        return m_data;
    }

    /**
     * Returns the raw data elements.
     */
    const T* data() const noexcept {
        return m_data;
    }

    /**
     * Returns the element given by the index.
     */
    const T operator[](std::initializer_list<size_t> index) const noexcept;

    /**
     * Returns the mutable element given by the index.
     */
    T& operator[](std::initializer_list<size_t> index) noexcept;

    /**
     * Returns a slice of tensor at the given index. The returned tensor
     * share the underlying data with this tensor, so the lifetime of the
     * returned tensor cannot exceed this tensor.
     */
    Tensor operator[](size_t index);

    /**
     * Equality test for two tensors.
     */
    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    // Tensor operators
    Tensor& operator+=(const Tensor& y);
    Tensor& operator+=(T v);
    Tensor& operator-=(const Tensor& y);
    Tensor& operator-=(T v);
    Tensor& operator*=(const Tensor& y);
    Tensor& operator*=(T v);
    Tensor& operator/=(const Tensor& y);
    Tensor& operator/=(T v);

    /**
     * Perform dot product on two matrices. A matrix is a 2-dimension tensor.
     */
    Tensor dot(const Tensor& y) const;

    /**
     * Transpose a matrix. A matrix is a 2-dimension tensor.
     */
    Tensor transpose() const;

    /**
     * Apply a function on tensor's elements.
     */
    template <typename F>
    Tensor& apply(F f);

    /**
     * Transform tensor's elements to a new tensor by applying the given function.
     * The element type may change during transformation.
     *
     * @param f the function to be applied.
     * @return the Tensor that contains transformed elements.
     */
    template <typename F, typename U = std::result_of_t<F(T)>>
    Tensor<U> transform(F f) const;

    /**
     * Transform tensor's elements to another tensor by applying the given function.
     * The two tensor must have the same shape.
     *
     * @param target the target tensor to store transformed data
     * @param f the function to apply the transformation
     */
    template <typename U, typename F>
    void transformTo(Tensor<U>& target, F f) const;

    /**
     * Casting element type.
     *
     * @tparam U the target element type
     * @return the Tensor with new element type.
     */
    template <typename U>
    Tensor<U> cast() const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        printRec(os, t.shape(), 0, t.data());
        return os;
    }

private:
    static const T* printRec(std::ostream& out, const Shape& shape, size_t level, const T* data);
};

template <typename T>
Tensor<T>::Tensor(Shape shape)
    : m_shape(std::move(shape))
{
    m_size = m_shape.size();
    m_alloc_data = std::make_unique<T[]>(m_size);
    m_data = m_alloc_data.get();
}

template <typename T>
Tensor<T>::Tensor(Shape shape, T* data)
    : m_shape(std::move(shape))
{
    m_data = data;
    m_size = m_shape.size();
}

template <typename T>
template <typename It>
Tensor<T>::Tensor(Shape shape, It begin, RequireInputIterator<It,T> end)
    : m_shape(std::move(shape))
{
    m_size = m_shape.size();
    assert(end - begin == m_size);
    m_alloc_data = std::make_unique<T[]>(m_size);
    m_data = m_alloc_data.get();
    std::copy(begin, end, m_data);
}

template <typename T>
Tensor<T>::Tensor(Shape shape, std::unique_ptr<T[]> data)
    : m_shape(std::move(shape)), m_alloc_data(std::move(data))
{
    m_data = m_alloc_data.get();
    m_size = m_shape.size();
}

template <typename T>
inline Tensor<T> Tensor<T>::wrap(Shape shape, T* data) {
    return Tensor(std::move(shape), data);
}

template <typename T>
template <typename F>
Tensor<T> Tensor<T>::build(Shape shape, F f) {
    auto size = shape.size();
    auto data = std::make_unique<T[]>(size);
    std::generate(data.get(), data.get() + size, f);
    return Tensor(std::move(shape), std::move(data));
}

template <typename T>
Tensor<T>::Tensor(const Tensor& t)
    : m_shape(t.m_shape), m_size(t.m_size)
{
    m_alloc_data = std::make_unique<T[]>(m_size);
    m_data = m_alloc_data.get();
    std::copy(t.data(), t.data() + t.size(), m_data);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& t) {
    m_shape = t.m_shape;
    if (m_size != t.m_size || m_alloc_data == nullptr) {
        m_size = t.m_size;
        m_alloc_data = std::make_unique<T[]>(m_size);
        m_data = m_alloc_data.get();
    }
    std::copy(t.data(), t.data() + t.size(), m_data);
    return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor&& t) noexcept
    : m_shape(std::move(t.m_shape)), m_size(t.m_size),
      m_data(t.m_data), m_alloc_data(std::move(t.m_alloc_data))
{
    t.m_data = nullptr;
    t.m_size = 0;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& t) noexcept {
    m_shape = std::move(t.m_shape);
    m_size = t.m_size;
    m_data = t.m_data;
    m_alloc_data = std::move(t.m_alloc_data);
    t.m_data = nullptr;
    t.m_size = 0;
    return *this;
}

template <typename T>
bool Tensor<T>::operator==(const Tensor& other) const {
    if (m_shape != other.m_shape)
        return false;
    return std::equal(data(), data()+size(), other.data());
}

template <typename T>
inline bool Tensor<T>::operator!=(const Tensor& other) const {
    return !(*this == other);
}

template <typename T>
inline const T Tensor<T>::operator[](std::initializer_list<size_t> index) const noexcept {
    return data()[m_shape.offset(index)];
}

template <typename T>
inline T& Tensor<T>::operator[](std::initializer_list<size_t> index) noexcept {
    return data()[m_shape.offset(index)];
}

template <typename T>
Tensor<T> Tensor<T>::operator[](size_t index) {
    assert(m_shape.rank() > 1);
    assert(index < m_shape[0]);

    auto slice_shape = m_shape.shrink();
    auto slice_size = slice_shape.size();
    auto slice_data = data() + index * slice_size;
    return wrap(slice_shape, slice_data);
}

template <typename T>
template <typename F>
inline Tensor<T>& Tensor<T>::apply(F f) {
    std::transform(data(), data()+size(), data(), f);
    return *this;
}

template <typename T>
template <typename F, typename U>
inline Tensor<U> Tensor<T>::transform(F f) const {
    Tensor<U> res(m_shape);
    std::transform(data(), data() + size(), res.data(), f);
    return res;
}

template <typename T>
template <typename U, typename F>
inline void Tensor<T>::transformTo(Tensor<U>& target, F f) const {
    assert(shape() == target.shape());
    std::transform(data(), data() + size(), target.data(), f);
}

template <typename T>
template <typename U>
inline Tensor<U> Tensor<T>::cast() const {
    return transform([](T x) { return static_cast<U>(x); });
}

/**
 * Perform binary operation on two tensors element and store the result into
 * the third tensor.
 */
template <typename T, typename F>
inline void transformTo(Tensor<T>& z, const Tensor<T>& x, const Tensor<T>& y, F f) {
    assert(x.shape() == y.shape() && y.shape() == z.shape());
    std::transform(x.data(), x.data() + x.size(), y.data(), z.data(), f);
}

/**
 * Perform binary operation on a tensor and a scalar value, store the
 * result into the third tensor.
 */
template <typename T, typename F>
void transformTo(Tensor<T>& z, const Tensor<T>& x, T y, F f) {
    assert(x.shape() == z.shape());
    auto px = x.data();
    auto pz = z.data();
    auto n  = x.size();
    for (size_t i = 0; i < n; i++, px++, pz++) {
        *pz = f(*px, y);
    }
}

/**
 * Perform binary operation on a scalar value and a tensor, store the
 * result into the third tensor.
 */
template <typename T, typename F>
void transformTo(Tensor<T>& z, T x, const Tensor<T>& y, F f) {
    assert(y.shape() == z.shape());
    auto py = y.data();
    auto pz = z.data();
    auto n  = y.size();
    for (size_t i = 0; i < n; i++, py++, pz++) {
        *pz = f(x, *py);
    }
}

/**
 * Perform binary operation on two tensors elements.
 */
template <typename T, typename F>
inline Tensor<T> transform(const Tensor<T>& x, const Tensor<T>& y, F f) {
    Tensor<T> z(x.shape());
    transformTo(z, x, y, f);
    return z;
}

/**
 * Perform binary operation on a tensor and a scalar value.
 */
template <typename T, typename F>
inline Tensor<T> transform(const Tensor<T>& x, T y, F f) {
    Tensor<T> z(x.shape());
    transformTo(z, x, y, f);
    return z;
}

/**
 * Perform binary operation on a scalar value and a tensor.
 */
template <typename T, typename F>
inline Tensor<T> transform(T x, const Tensor<T>&y, F f) {
    Tensor<T> z(y.shape());
    transformTo(z, x, y, f);
    return z;

}

#define DEFINE_OPERATOR(op, fn) \
    template <typename T> \
    inline Tensor<T>& Tensor<T>::operator op##=(const Tensor<T>& y) { \
        kneron::model::transformTo(*this, *this, y, fn<T>()); \
        return *this; \
    } \
    template <typename T> \
    inline Tensor<T>& Tensor<T>::operator op##=(T y) { \
        kneron::model::transformTo(*this, *this, y, fn<T>()); \
        return *this; \
    } \
    template <typename T> \
    inline Tensor<T> operator op(const Tensor<T>& x, const Tensor<T>& y) { \
        return transform(x, y, fn<T>()); \
    } \
    template <typename T> \
    inline Tensor<T> operator op(const Tensor<T>& x, T y) { \
        return transform(x, y, fn<T>()); \
    } \
    template <typename T> \
    inline Tensor<T> operator op(T x, const Tensor<T>& y) { \
        return transform(x, y, fn<T>()); \
    }

DEFINE_OPERATOR(+, std::plus)
DEFINE_OPERATOR(-, std::minus)
DEFINE_OPERATOR(*, std::multiplies)
DEFINE_OPERATOR(/, std::divides)

#undef DEFINE_OPERATOR

template <typename T>
Tensor<T> Tensor<T>::dot(const Tensor& y) const {
    assert(is_matrix() && y.is_matrix());

    auto n = m_shape[0];
    auto p = m_shape[1];
    auto m = y.m_shape[1];
    assert(p == y.m_shape[0]);

    Tensor z({n, m});
    auto px = data();
    auto py = y.data();
    auto pz = z.data();
    int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            T v = T();
            for (k = 0; k < p; k++)
                v += px[i * p + k] * py[k * m + j];
            pz[i * m + j] = v;
        }
    }

    return z;
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
    assert(is_matrix());

    auto n = m_shape[0];
    auto m = m_shape[1];

    Tensor y({m, n});
    auto px = data();
    auto py = y.data();
    int i, j;

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            *py++ = px[j * m + i];
    return y;
}

template <typename T>
const T* Tensor<T>::printRec(std::ostream& out, const Shape& shape, size_t level, const T* data) {
    auto d = shape[level];

    out << '[';
    if (level == shape.rank()-1) {
        // last level, printing data
        for (int i = 0; i < d; i++) {
            out << *data++;
            if (i < d-1)
                out << ", ";
        }
    } else {
        // intermediate levels, recursive
        for (int i = 0; i < d; i++) {
            data = printRec(out, shape, level+1, data);
            if (i < d-1) {
                out << ',' << std::endl;
                for (int j = 0; j <= level; j++)
                    out << ' ';
            }
        }
    }
    out << "]";

    return data;
}

} // namespace kneron::model

#endif //KNERON_TENSOR_H
