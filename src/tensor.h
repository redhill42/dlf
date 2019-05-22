#ifndef KNERON_TENSOR_H
#define KNERON_TENSOR_H

#include <vector>
#include <complex>
#include <random>
#include <iostream>

#include "concurrent.h"
#include "os_blas.h"

namespace kneron::model {

//==-------------------------------------------------------------------------
// Tensor declaration
//==-------------------------------------------------------------------------

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
    size_t extent(size_t dim) const noexcept {
        return m_dims[dim];
    }

    /**
     * Returns number of elements in a given dimension.
     */
    size_t operator[](size_t dim) const noexcept {
        return m_dims[dim];
    }

    /**
     * Return pair of extents if this shape represents a matrix.
     */
    std::pair<size_t,size_t> extent() const noexcept {
        assert(rank() == 2);
        return std::pair(extent(0), extent(1));
    }

    /**
     * Change dimensions of this shape. The new shape must compatible to
     * this shape.
     */
    bool reshape(Shape newshape);

    /**
     * Compare two shapes for equality.
     */
    bool operator==(const Shape& other) const {
        return m_dims == other.m_dims;
    }

    /**
     * Compare two shapes for non-equality.
     */
    bool operator!=(const Shape& other) const {
        return m_dims != other.m_dims;
    }

    /**
     * Shrink one level of dimensions.
     */
    Shape shrink() const {
        return Shape(std::vector(std::next(m_dims.begin()), m_dims.end()));
    }

    /**
     * Returns the data size defined by this shape.
     */
    size_t size() const noexcept;

    /**
     * Return the data offset for the given index.
     */
    size_t offset(std::initializer_list<size_t> index) const noexcept;

    /**
     * Returns the next index within this shape.
     *
     * @return true if next index is available
     */
    bool next(std::vector<size_t>& index) const noexcept;
};

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

    void init() {
        m_size = m_shape.size();
        m_alloc_data = std::make_unique<T[]>(m_size);
        m_data = m_alloc_data.get();
    }

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

private: // Concepts
    template <typename InputIterator>
    using RequireInputIterator =
        std::enable_if_t<
            std::is_convertible_v<
                typename std::iterator_traits<InputIterator>::iterator_category,
                std::input_iterator_tag> &&
            std::is_constructible_v<
                T, typename std::iterator_traits<InputIterator>::reference>,
            InputIterator>;

    template <typename... Args>
    using RequireIndexes =
        std::enable_if_t<std::conjunction_v<std::is_convertible<Args, size_t>...>>;

private:
    /**
     * Construct a tensor with given dimension and wrapped data. This constructor
     * can only be called from wrap() function.
     *
     * @param shape the tensor dimensions
     * @param data the tensor data
     */
    Tensor(Shape shape, T* data);

public: // Constructors
    /**
     * Construct a 0-dimensional tensor.
     */
    Tensor() = default;

    /**
     * Construct a tensor with given dimensions.
     *
     * @param shape the tensor shape
     */
    explicit Tensor(const Shape& shape); // NOLINT(modernize-pass-by-value)
    explicit Tensor(Shape&& shape);

    /**
     * Construct a tensor with input iterator denoted by [begin,end).
     *
     * @param shape the tensor dimensions
     * @param begin the start of input iterator
     * @param end the end of input iterator
     */
    template <typename It>
    Tensor(Shape shape, It begin, RequireInputIterator<It> end);

    /**
     * Construct a tensor with an initializer list.
     *
     * @param shape the tensor dimensions
     * @param init the initializer list
     */
    Tensor(Shape shape, std::initializer_list<T> init);

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
     * Build the tensor with given generator function.
     *
     * @param shape the tensor dimension
     * @param f the generator function
     * @return the generated tensor
     */
    template <typename F>
    static Tensor build(Shape shape, F f);

    /**
     * Create an n by n identity matrix.
     *
     * @param n the matrix dimension
     * @param value the identity value
     * @return the identity matrix
     */
    static Tensor identity(size_t n, const T& value = T{1});

    /**
     * Create a tensor with values starting from n.
     *
     * @param shape the tensor dimension
     * @param n the starting value in the tensor data.
     */
    static Tensor range(Shape shape, T n);

    /**
     * Create a tensor filled with random data.
     *
     * @param shape the tensor dimension
     * @param low the lowest random value
     * @param high the highest random value
     * @return a tensor that filled with random data.
     */
    static Tensor random(Shape shape, const T& low, const T& high);

    // Copy and move constructors/assignments.
    Tensor(const Tensor& t);
    Tensor& operator=(const Tensor& t);
    Tensor(Tensor&& t) noexcept;
    Tensor& operator=(Tensor&& t) noexcept;

public: // Attributes
    /**
     * Returns the shape of this tensor.
     */
    const Shape& shape() const noexcept {
        return m_shape;
    }

    /**
     * Reshape the tensor without changing tensor data. The new shape
     * should be compatible with the original shape. At most one
     * dimension of the new shape can be -1. In this case, the
     * actual dimension value is inferred from the size of the tensor
     * and the remaining dimensions.
     *
     * @param newshape specifies the new shape.
     * @return true if shape changed, false if new shape is not
     * compatible with current shape.
     */
    bool reshape(Shape newshape) {
        return m_shape.reshape(std::move(newshape));
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
     * Returns true if this tensor is a square matrix.
     */
    bool is_square() const noexcept {
        return is_matrix() && m_shape[0] == m_shape[1];
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
    template <typename... Args, typename = RequireIndexes<Args...>>
    const T& operator()(Args... args) const noexcept;

    /**
     * Returns the mutable element given by the index.
     */
    template <typename... Args, typename = RequireIndexes<Args...>>
    T& operator()(Args... args) noexcept;

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

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        if (!t.empty())
            printRec(os, t.shape(), 0, t.data());
        return os;
    }

public: // Operators
#define DECLARE_OPERATOR(op) \
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U,T>>> \
    Tensor& operator op(const Tensor<U>& y); \
    Tensor& operator op(const T& b);

    DECLARE_OPERATOR(+=)
    DECLARE_OPERATOR(-=)
    DECLARE_OPERATOR(*=)
    DECLARE_OPERATOR(/=)
#undef DECLARE_OPERATOR

public: // Transformations
    /**
     * Transform tensor's elements by applying a unary function on tensor's elements.
     *
     * @param f the unary function.
     * @return *this (useful for chained operation)
     */
    template <typename F>
    Tensor& apply(F f);

    /**
     * Apply the function pointer on tensor's elements. This is a workaround for
     * overloaded function deduction issue.
     *
     * @param f the function pointer
     * @return *this (useful for chained operation)
     */
    Tensor& apply(T(*f)(T));

    /**
     * Transform two tensor's elements by applying a binary function.
     *
     * @param y another tensor involved in apply.
     * @param f the binary function
     * @return *this (useful for chained operation)
     */
    template <typename U, typename F>
    Tensor& apply(const Tensor<U>& y, F f);

    /**
     * Transform a tensor to a new tensor by applying the given unary function
     * on tensor's elements.
     *
     * @param f the unary function
     * @return the transformed tensor
     */
    template <typename F, typename U = std::invoke_result_t<F,T>>
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
     * Transform two tensor's elements to a new tensor by applying the given
     * binary function. The element type may change during transformation.
     *
     * @param y another tensor involved in transformation
     * @param f the function to be applied.
     * @return the Tensor that contains transformed elements.
     */
    template <typename U, typename F, typename W = std::invoke_result_t<F,T,U>>
    Tensor<W> transform(const Tensor<U>& y, F f) const;

    /**
     * Transform two tensor's elements to another tensor by applying the given
     * binary function. The two tensors must have the same shape.
     *
     * @param target the target tensor to store transformed data.
     * @param y another tensor involved in transformation.
     * @param f the function to be applied.
     */
    template <typename U, typename W, typename F>
    void transformTo(Tensor<W>& z, const Tensor<U>& y, F f) const;

    // Rvalue optimization for transformations.
    template <typename F, typename = std::enable_if_t<std::is_same_v<std::invoke_result_t<F,T>,T>>>
    Tensor<T> transform(F f) &&;
    template <typename F, typename = std::enable_if_t<std::is_same_v<std::invoke_result_t<F,T,T>,T>>>
    Tensor<T> transform(const Tensor<T>& y, F f) &&;
    template <typename F, typename = std::enable_if_t<std::is_same_v<std::invoke_result_t<F,T,T>,T>>>
    Tensor<T> transform(Tensor<T>&& y, F f) const &;

    /**
     * Casting element type.
     *
     * @tparam U the target element type
     * @return the Tensor with new element type.
     */
    template <typename U>
    Tensor<U> cast() const;

private: // Implementation
    static const T* printRec(std::ostream& out, const Shape& shape, size_t level, const T* data);
};

//==-------------------------------------------------------------------------
// Tensor constructors
//==-------------------------------------------------------------------------

template <typename T>
Tensor<T>::Tensor(const Shape& shape) // NOLINT(modernize-pass-by-value)
    : m_shape(shape)
{
    init();
}

template <typename T>
Tensor<T>::Tensor(Shape&& shape)
    : m_shape(std::move(shape))
{
    init();
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
Tensor<T>::Tensor(Shape shape, It begin, RequireInputIterator<It> end)
    : m_shape(std::move(shape))
{
    init();
    assert(std::distance(begin, end) == m_size);
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
Tensor<T>::Tensor(Shape shape, std::initializer_list<T> il)
    : m_shape(std::move(shape))
{
    init();
    assert(m_size == il.size());
    std::copy(il.begin(), il.end(), m_data);
}

template <typename T>
inline Tensor<T> Tensor<T>::wrap(Shape shape, T* data) {
    return Tensor(std::move(shape), data);
}

template <typename T>
template <typename F>
inline Tensor<T> Tensor<T>::build(Shape shape, F f) { // NOLINT(performance-unnecessary-value-param)
    Tensor<T> res(std::move(shape));
    std::generate(res.begin(), res.end(), f);
    return res;
}

template <typename T>
Tensor<T> Tensor<T>::identity(size_t n, const T& value) {
    Tensor<T> res({n, n});
    T* p = res.data();
    for (size_t i = 0; i < n; i++, p += n+1)
        *p = value;
    return res;
}

template <typename T>
Tensor<T> Tensor<T>::range(Shape shape, T n) {
    Tensor<T> res(std::move(shape));
    T* p = res.data();
    for (size_t k = res.size(); k-- != 0; )
        *p++ = n++;
    return res;
}

template <typename T>
Tensor<T> Tensor<T>::random(Shape shape, const T& low, const T& high) {
    static_assert(std::is_arithmetic_v<T>);

    std::random_device r;
    std::default_random_engine eng(r());

    if constexpr (std::is_integral_v<T>) {
        auto rand = std::bind(std::uniform_int_distribution<T>(low, high), eng);
        return build(std::move(shape), rand);
    }

    if constexpr (std::is_floating_point_v<T>) {
        auto rand = std::bind(std::uniform_real_distribution<T>(low, high), eng);
        return build(std::move(shape), rand);
    }
}

template <typename T>
Tensor<T>::Tensor(const Tensor& t)
    : m_shape(t.m_shape), m_size(t.m_size)
{
    m_alloc_data = std::make_unique<T[]>(m_size);
    m_data = m_alloc_data.get();
    std::copy(t.begin(), t.end(), m_data);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& t) {
    m_shape = t.m_shape;
    if (m_size != t.m_size || m_alloc_data == nullptr) {
        m_size = t.m_size;
        m_alloc_data = std::make_unique<T[]>(m_size);
        m_data = m_alloc_data.get();
    }
    std::copy(t.begin(), t.end(), m_data);
    return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor&& t) noexcept
    : m_shape(std::move(t.m_shape)),
      m_size(std::exchange(t.m_size, 0)),
      m_data(std::exchange(t.m_data, nullptr)),
      m_alloc_data(std::move(t.m_alloc_data))
{
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& t) noexcept {
    m_shape = std::move(t.m_shape);
    m_size = std::exchange(t.m_size, 0);
    m_data = std::exchange(t.m_data, nullptr);
    m_alloc_data = std::move(t.m_alloc_data);
    return *this;
}

//==-------------------------------------------------------------------------
// Tensor attributes
//==-------------------------------------------------------------------------

template <typename T>
inline bool Tensor<T>::operator==(const Tensor& other) const {
    return shape() == other.shape() && std::equal(begin(), end(), other.begin());
}

template <typename T>
inline bool Tensor<T>::operator!=(const Tensor& other) const {
    return !(*this == other);
}

template<typename T>
template<typename... Args, typename>
inline const T& Tensor<T>::operator()(Args... args) const noexcept {
    return data()[shape().offset({size_t(args)...})];
}

template<typename T>
template<typename... Args, typename>
inline T& Tensor<T>::operator()(Args... args) noexcept {
    return data()[shape().offset({size_t(args)...})];
}

template <typename T>
Tensor<T> Tensor<T>::operator[](size_t index) {
    assert(m_shape.rank() > 1);
    assert(index < m_shape[0]);

    auto slice_shape = m_shape.shrink();
    auto slice_size = slice_shape.size();
    auto slice_data = data() + index * slice_size;
    return wrap(std::move(slice_shape), slice_data);
}

template <typename T>
const T* Tensor<T>::printRec(std::ostream& out, const Shape& shape, size_t level, const T* data) {
    auto d = shape.extent(level);

    out << '[';
    if (level == shape.rank()-1) {
        // last level, printing data
        for (int i = 0; i < d; i++) {
            out << *data++;
            if (i < d-1)
                out << ',';
        }
    } else {
        // intermediate levels, recursive
        for (int i = 0; i < d; i++) {
            data = printRec(out, shape, level+1, data);
            if (i < d-1)
                out << ',';
        }
    }
    out << ']';

    return data;
}

//==-------------------------------------------------------------------------
// Tensor transformations
//==-------------------------------------------------------------------------

template <typename T>
template <typename F>
inline Tensor<T>& Tensor<T>::apply(F f) {
    kneron::concurrent::parallel_transform(begin(), end(), begin(), f);
    return *this;
}

template <typename T>
inline Tensor<T>& Tensor<T>::apply(T(*f)(T)) {
    kneron::concurrent::parallel_transform(begin(), end(), begin(), f);
    return *this;
}

template <typename T>
template <typename U, typename F>
inline Tensor<T>& Tensor<T>::apply(const Tensor<U>& y, F f) {
    assert(shape() == y.shape());
    kneron::concurrent::parallel_transform(begin(), end(), y.begin(), begin(), f);
    return *this;
}

template <typename T>
template <typename F, typename U>
inline Tensor<U> Tensor<T>::transform(F f) const {
    Tensor<U> res(shape());
    kneron::concurrent::parallel_transform(begin(), end(), res.begin(), f);
    return res;
}

template <typename T>
template <typename U, typename F>
inline void Tensor<T>::transformTo(Tensor<U>& target, F f) const {
    assert(shape() == target.shape());
    kneron::concurrent::parallel_transform(begin(), end(), target.begin(), f);
}

template <typename T>
template <typename U, typename F, typename W>
inline Tensor<W> Tensor<T>::transform(const Tensor<U>& y, F f) const {
    assert(shape() == y.shape());
    Tensor<W> z(shape());
    kneron::concurrent::parallel_transform(begin(), end(), y.begin(), z.begin(), f);
    return z;
}

template <typename T>
template <typename U, typename W, typename F>
inline void Tensor<T>::transformTo(Tensor<W>& z, const Tensor<U>& y, F f) const {
    assert(shape() == y.shape() && shape() == z.shape());
    kneron::concurrent::parallel_transform(begin(), end(), y.begin(), z.begin(), f);
}

template <typename T>
template <typename F, typename>
inline Tensor<T> Tensor<T>::transform(F f) && {
    return std::move(apply(f));
}

template <typename T>
template <typename F, typename>
inline Tensor<T> Tensor<T>::transform(const Tensor<T> &y, F f) && {
    return std::move(apply(y, f));
}

template <typename T>
template <typename F, typename>
inline Tensor<T> Tensor<T>::transform(Tensor<T>&& y, F f) const & {
    assert(shape() == y.shape());
    kneron::concurrent::parallel_transform(begin(), end(), y.begin(), y.begin(), f);
    return std::move(y);
}

template <typename T>
template <typename U>
inline Tensor<U> Tensor<T>::cast() const {
    return transform([](const T& x) { return static_cast<U>(x); });
}

//==-------------------------------------------------------------------------
// Tensor operators
//==-------------------------------------------------------------------------

#define DEFINE_OPERATOR(op) \
    template <typename T> \
    template <typename U, typename> \
    inline Tensor<T>& Tensor<T>::operator op##=(const Tensor<U>& y) { \
        return apply(y, [](const T& a, const U& b) {return a op b;}); \
    } \
    template <typename T> \
    inline Tensor<T>& Tensor<T>::operator op##=(const T& b) { \
        return apply([&b](const T& a) {return a op b;}); \
    } \
    template <typename T, typename U, typename W = std::common_type_t<T,U>> \
    inline Tensor<W> operator op(const Tensor<T>& x, const Tensor<U>& y) { \
        return x.transform(y, [](const T& a, const U& b) -> W {return a op b;}); \
    } \
    template <typename T, typename U, typename W = std::common_type_t<T,U>> \
    inline Tensor<W> operator op(const Tensor<T>& x, const U& b) { \
        return x.transform([&b](const T& a) -> W {return a op b;}); \
    } \
    template <typename T, typename U, typename W = std::common_type_t<T,U>> \
    inline Tensor<W> operator op(const T& a, const Tensor<U>& y) { \
        return y.transform([&a](const U& b) -> W {return a op b;}); \
    } \
    /* rvalue optimization */ \
    template <typename T> \
    inline Tensor<T> operator op(Tensor<T>&& x, const Tensor<T>& y) { \
        return std::move(x.apply(y, [](const T& a, const T& b) {return a op b;})); \
    } \
    template <typename T> \
    inline Tensor<T> operator op(const Tensor<T>& x, Tensor<T>&& y) { \
        return std::move(y.apply(x, [](const T& b, const T& a) {return a op b;})); \
    } \
    template <typename T> \
    inline Tensor<T> operator op(Tensor<T>&& x, Tensor<T>&& y) { \
        return std::move(x.apply(y, [](const T& a, const T& b) {return a op b;})); \
    } \
    template <typename T> \
    inline Tensor<T> operator op(Tensor<T>&& x, const T& b) { \
        return std::move(x.apply([&b](const T& a) {return a op b;})); \
    } \
    template <typename T> \
    inline Tensor<T> operator op(const T& a, Tensor<T>&& y) { \
        return std::move(y.apply([&a](const T& b) {return a op b;})); \
    }

DEFINE_OPERATOR(+)
DEFINE_OPERATOR(-)
DEFINE_OPERATOR(*)
DEFINE_OPERATOR(/)

#undef DEFINE_OPERATOR

#define DEFINE_BLAS_OPERATOR(op, T, alpha, beta) \
    template <> template <> \
    inline Tensor<T>& Tensor<T>::operator op##=(const Tensor<T>& rhs) { \
        assert(shape() == rhs.shape()); \
        cblas::axpby(size(), T(beta), rhs.data(), 1, T(alpha), data(), 1); \
        return *this; \
    } \
    template <> \
    inline Tensor<T> operator op(const Tensor<T>& lhs, const Tensor<T>& rhs) { \
        assert(lhs.shape() == rhs.shape()); \
        Tensor<T> res(lhs.shape()); \
        std::copy(rhs.begin(), rhs.end(), res.begin()); \
        cblas::axpby(res.size(), T(alpha), lhs.data(), 1, T(beta), res.data(), 1); \
        return res; \
    } \
    template <> \
    inline Tensor<T> operator op(Tensor<T>&& lhs, const Tensor<T>& rhs) { \
        assert(lhs.shape() == rhs.shape()); \
        cblas::axpby(lhs.size(), T(beta), rhs.data(), 1, T(alpha), lhs.data(), 1); \
        return std::move(lhs); \
    } \
    template <> \
    inline Tensor<T> operator op(const Tensor<T>& lhs, Tensor<T>&& rhs) { \
        assert(lhs.shape() == rhs.shape()); \
        cblas::axpby(rhs.size(), T(alpha), lhs.data(), 1, T(beta), rhs.data(), 1); \
        return std::move(rhs); \
    } \
    template <> \
    inline Tensor<T> operator op(Tensor<T>&& lhs, Tensor<T>&& rhs) { \
        assert(lhs.shape() == rhs.shape()); \
        cblas::axpby(lhs.size(), T(alpha), lhs.data(), 1, T(beta), rhs.data(), 1); \
        return std::move(rhs); \
    }

DEFINE_BLAS_OPERATOR(+, float, 1.0, 1.0)
DEFINE_BLAS_OPERATOR(-, float, 1.0, -1.0)
DEFINE_BLAS_OPERATOR(+, double, 1.0, 1.0)
DEFINE_BLAS_OPERATOR(-, double, 1.0, -1.0)
DEFINE_BLAS_OPERATOR(+, std::complex<float>, 1.0, 1.0)
DEFINE_BLAS_OPERATOR(-, std::complex<float>, 1.0, -1.0)
DEFINE_BLAS_OPERATOR(+, std::complex<double>, 1.0, 1.0)
DEFINE_BLAS_OPERATOR(-, std::complex<double>, 1.0, -1.0)

#undef DEFINE_BLAS_OPERATOR

template <typename T>
inline Tensor<T> operator-(const Tensor<T>& x) {
    return x.transform([](const T& a){return -a;});
}

template <typename T>
inline Tensor<T> operator-(Tensor<T>&& x) {
    return std::move(x.apply([](const T& a){return -a;}));
}

//==-------------------------------------------------------------------------
// Tensor operations
//==-------------------------------------------------------------------------

namespace impl {

template <typename T>
T vector_dot_vector(size_t n, const T* A, const T* B) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        return cblas::dot(n, A, 1, B, 1);
    }

    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n, GRAINSIZE),
        T{},
        [&](auto&& r, T sum) {
            auto px = A + r.begin();
            auto py = B + r.begin();
            for (size_t k = r.size(); k-- != 0; )
                sum += *px++ * *py++;
            return sum;
        },
        std::plus());
}

template <typename T>
void matrix_dot_vector(size_t m, size_t n, const T* A, const T* B, T* C) {
    if constexpr (cblas::IsBlasType<T>) {
        cblas::gemv(CblasRowMajor, CblasNoTrans, m, n, T(1), A, n, B, 1, T(0), C, 1);
        return;
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, m, GRAINSIZE), [&](auto&& r) {
        auto px = A + r.begin() * n;
        auto pz = C + r.begin();
        for (size_t k = r.size(); k-- != 0; ) {
            auto py = B;
            T v{};
            for (size_t j = 0; j < n; j++)
                v += *px++ * *py++;
            *pz++ = std::move(v);
        }
    });
}

template <typename T>
void matrix_dot_matrix(size_t m, size_t k, size_t n, const T* A, const T* B, T* C) {
    if constexpr (cblas::IsBlasType<T>) {
        cblas::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, T(1), A, k, B, n, T(0), C, n);
        return;
    }

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [&](auto &&r) {
        for (size_t i = r.rows().begin(); i != r.rows().end(); i++) {
            for (size_t j = r.cols().begin(); j != r.cols().end(); j++) {
                T v{};
                for (size_t t = 0; t < k; t++)
                    v += A[i * k + t] * B[t * n + j];
                C[i * n + j] = std::move(v);
            }
        }
    });
}

} // namespace impl

/**
 * Perform inner product on two tensors. The tensors must be vector
 * or matrix and have compatible dimensions.
 */
template <typename T>
Tensor<T>& inner(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>* C) {
    assert(C != &A && C != &B);
    if (A.is_vector() && B.is_vector()) {
        auto n = A.shape()[0];
        assert(n == B.shape()[0]);
        assert(C->is_vector() && 1 == C->shape()[0]);
        *C->data() = impl::vector_dot_vector(n, A.data(), B.data());
    } else if (A.is_vector() && B.is_matrix()) {
        auto [k, n] = B.shape().extent();
        assert(k == A.shape()[0]);
        assert(C->is_vector() && n == C->shape()[0]);
        impl::matrix_dot_matrix(1, k, n, A.data(), B.data(), C->data());
    } else if (A.is_matrix() && B.is_vector()) {
        auto [m, n] = A.shape().extent();
        assert(n == B.shape()[0]);
        assert(C->is_vector() && m == C->shape()[0]);
        impl::matrix_dot_vector(m, n, A.data(), B.data(), C->data());
    } else if (A.is_matrix() && B.is_matrix()) {
        auto [m, k] = A.shape().extent();
        auto [p, n] = B.shape().extent();
        assert(k == p);
        assert(C->shape() == Shape({m, n}));
        impl::matrix_dot_matrix(m, k, n, A.data(), B.data(), C->data());
    } else {
        assert(false);
    }
    return *C;
}

template <typename T>
Tensor<T> inner(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.is_vector() && B.is_vector()) {
        auto n = A.shape()[0];
        assert(n == B.shape()[0]);
        return Tensor<T>({1}, {impl::vector_dot_vector(n, A.data(), B.data())});
    } else if (A.is_vector() && B.is_matrix()) {
        auto [k, n] = B.shape().extent();
        assert(k == A.shape()[0]);
        Tensor<T> C({n});
        impl::matrix_dot_matrix(1, k, n, A.data(), B.data(), C.data());
        return C;
    } else if (A.is_matrix() && B.is_vector()) {
        auto [m, n] = A.shape().extent();
        assert(n == B.shape()[0]);
        Tensor<T> C({m});
        impl::matrix_dot_vector(m, n, A.data(), B.data(), C.data());
        return C;
    } else if (A.is_matrix() && B.is_matrix()) {
        auto [m, k] = A.shape().extent();
        auto [p, n] = B.shape().extent();
        assert(k == p);
        Tensor<T> C({m, n});
        impl::matrix_dot_matrix(m, k, n, A.data(), B.data(), C.data());
        return C;
    } else {
        assert(false);
        return Tensor<T>();
    }
}

template <typename T>
Tensor<T> pow(const Tensor<T>& x, long n) {
    assert(x.is_square() && n >= 0);
    if (n == 0)
        return Tensor<T>::identity(x.shape()[0]);
    n--;

    auto A = x, B = x, t = x;
    while (n > 0) {
        if (n & 1)
            std::swap(B, inner(A, B, &t));
        std::swap(A, inner(A, A, &t));
        n >>= 1;
    }
    return B;
}

/**
 * General matrix multiplication.
 */
template <typename T>
void gemm(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>* C,
          const T& alpha, const T& beta, bool transA = false, bool transB = false)
{
    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto [m, k] = A.shape().extent();
    if (transA)
        std::swap(m, k);

    auto [p, n] = B.shape().extent();
    if (transB)
        std::swap(p, n);

    assert(k == p);
    assert(C->shape() == Shape({m, n}));

    if constexpr (cblas::IsBlasType<T>) {
        cblas::gemm(CblasRowMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    m, n, k,
                    alpha,
                    A.data(), A.shape()[1],
                    B.data(), B.shape()[1],
                    beta,
                    C->data(), n);
        return;
    }

    if (alpha == T(0)) {
        *C *= beta;
        return;
    }

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [&, m=m, k=k, n=n](auto &&r) {
        size_t incX = transA ? m : 1;
        size_t incY = transB ? 1 : n;
        for (size_t i = r.rows().begin(); i != r.rows().end(); i++) {
            T* pz = &C->data()[i * n + r.cols().begin()];
            for (size_t j = r.cols().begin(); j != r.cols().end(); j++) {
                const T* px = A.data() + (transA ? i : i*k);
                const T* py = B.data() + (transB ? j*k : j);
                T v = *pz * beta;
                for (size_t t = 0; t < k; t++) {
                    v += alpha * *px * *py;
                    px += incX;
                    py += incY;
                }
                *pz++ = std::move(v);
            }
        }
    });
}

template <typename T>
void gemm(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C, Tensor<T>* R,
          const T& alpha, const T& beta, bool transA = false, bool transB = false)
{
    std::copy(C.begin(), C.end(), R->begin());
    gemm(A, B, R, alpha, beta, transA, transB);
}

template <typename T>
Tensor<T> gemm(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& C,
               const T& alpha, const T& beta, bool transA = false, bool transB = false)
{
    Tensor<T> R = C;
    gemm(A, B, &R, alpha, beta, transA, transB);
    return R;
}

namespace impl {

// Simple case: transpose with copy
template <typename T>
void copy_transpose(size_t r, size_t c, const T* src, T* dst) {
#if HAS_MKL
    if constexpr (cblas::IsBlasType<T>) {
        mkl::omatcopy('R', 'T', r, c, T(1), src, c, dst, r);
        return;
    }
#endif

    tbb::parallel_for(tbb::blocked_range<size_t>(0, c, GRAINSIZE), [=](auto&& rr) {
        T* py = dst + rr.begin()*r;
        for (size_t i = rr.begin(); i != rr.end(); i++) {
            auto px = src + i;
            for (size_t j = 0; j < r; j++, px += c)
                *py++ = *px;
        }
    });
}

// Easy case: in-place transpose a square matrix
template <typename T>
void square_transpose(size_t n, T* A) {
#if HAS_MKL
    if constexpr (cblas::IsBlasType<T>) {
        mkl::imatcopy('R', 'T', n, n, T(1), A, n, n);
        return;
    }
#endif

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n, GRAINSIZE), [=](auto&& r) {
        T* px = A;
        for (size_t i = r.begin(); i != r.end(); i++) {
            for (size_t j = i+1; j < n; j++) {
                std::swap(px[i*n+j], px[j*n+i]);
            }
        }
    });
}

// Hard case: in-place transpose a non-square matrix
// https://en.wikipedia.org/wiki/In-place_matrix_transposition
template <typename T>
void inplace_transpose(size_t r, size_t c, T* A) {
#if HAS_MKL
    if constexpr (cblas::IsBlasType<T>) {
        mkl::imatcopy('R', 'T', r, c, T(1), A, c, r);
        return;
    }
#endif

    // naive implementation
    Tensor<T> t({r, c}, A, A+r*c);
    copy_transpose(r, c, t.data(), A);
}

} // namespace impl

/**
 * Transpose a matrix to target matrix.
 *
 * @param from the matrix to be transposed
 * @param to the transposed matrix
 */
template <typename T>
void transpose(const Tensor<T>& from, Tensor<T>* to) {
    assert(from.is_matrix() && to->is_matrix());
    auto [r, c] = from.shape().extent();
    if (&from == to) {
        if (r == c) {
            impl::square_transpose(r, to->data());
        } else {
            to->reshape({c, r});
            impl::inplace_transpose(r, c, to->data());
        }
    } else {
        assert(to->shape() == Shape({c, r}));
        impl::copy_transpose(r, c, from.data(), to->data());
    }
}

/**
 * Transpose a matrix.
 *
 * @param A the matrix to be transposed
 * @return the transposed matrix
 */
template <typename T>
Tensor<T> transpose(const Tensor<T>& A) {
    assert(A.is_matrix());
    auto [r, c] = A.shape().extent();
    Tensor<T> B = Tensor<T>({c, r});
    impl::copy_transpose(r, c, A.data(), B.data());
    return B;
}

} // namespace kneron::model

#endif //KNERON_TENSOR_H
