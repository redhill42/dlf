#pragma once

#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <iostream>

#include "utility.h"
#include "tensor/shape.h"
#include "parallel.h"
#include "os_blas.h"

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor declaration
//==-------------------------------------------------------------------------

/**
 * Tensor is a geometric object that maps in a multi-linear manner geometric
 * vectors, scalars, and other tensors to a resulting tensor.
 *
 * @tparam T the data type of the tensor.
 */
template <typename T>
class Tensor : public Shaped {
    T* m_data = nullptr;
    std::unique_ptr<T[]> m_alloc_data;

    void init() {
        m_alloc_data = std::make_unique<T[]>(size());
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
            std::is_convertible<
                typename std::iterator_traits<InputIterator>::iterator_category,
                std::input_iterator_tag>::value &&
            std::is_constructible<
                T, typename std::iterator_traits<InputIterator>::reference>::value,
            InputIterator>;

    template <typename... Args>
    using RequireIndexes =
        std::enable_if_t<cxx::conjunction<std::is_convertible<Args, size_t>...>::value>;

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
    explicit Tensor(Shape shape);

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
     * @param step the increment step
     */
    static Tensor range(Shape shape, T n, T step = T{1});

    /**
     * Create a tensor filled with random data.
     *
     * @param shape the tensor dimension
     * @param low the lowest random value
     * @param high the highest random value
     * @return a tensor that filled with random data.
     */
    static Tensor random(Shape shape, T low, T high);

    // Copy and move constructors/assignments.
    Tensor(const Tensor& t);
    Tensor& operator=(const Tensor& t);
    Tensor(Tensor&& t) noexcept;
    Tensor& operator=(Tensor&& t) noexcept;

public: // Attributes
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
    template <typename U, typename = std::enable_if_t<std::is_convertible<U,T>::value>> \
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
    template <typename F, typename U = cxx::invoke_result_t<F,T>>
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
    template <typename U, typename F, typename W = cxx::invoke_result_t<F,T,U>>
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
    template <typename F, typename = std::enable_if_t<std::is_same<cxx::invoke_result_t<F,T>,T>::value>>
    Tensor<T> transform(F f) &&;
    template <typename F, typename = std::enable_if_t<std::is_same<cxx::invoke_result_t<F,T,T>,T>::value>>
    Tensor<T> transform(const Tensor<T>& y, F f) &&;
    template <typename F, typename = std::enable_if_t<std::is_same<cxx::invoke_result_t<F,T,T>,T>::value>>
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
Tensor<T>::Tensor(Shape shape)
    : Shaped(std::move(shape))
{
    init();
}

template <typename T>
Tensor<T>::Tensor(Shape shape, T* data)
    : Shaped(std::move(shape))
{
    m_data = data;
}

template <typename T>
template <typename It>
Tensor<T>::Tensor(Shape shape, It begin, RequireInputIterator<It> end)
    : Shaped(std::move(shape))
{
    init();
    assert(std::distance(begin, end) == size());
    std::copy(begin, end, m_data);
}

template <typename T>
Tensor<T>::Tensor(Shape shape, std::unique_ptr<T[]> data)
    : Shaped(std::move(shape)), m_alloc_data(std::move(data))
{
    m_data = m_alloc_data.get();
}

template <typename T>
Tensor<T>::Tensor(Shape shape, std::initializer_list<T> il)
    : Shaped(std::move(shape))
{
    init();
    assert(size() == il.size());
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
Tensor<T> Tensor<T>::range(Shape shape, T n, T step) {
    Tensor<T> res(std::move(shape));
    T* p = res.data();
    for (size_t k = res.size(); k-- != 0; n += step)
        *p++ = n;
    return res;
}

template <typename T>
Tensor<T> Tensor<T>::random(Shape shape, T low, T high) {
    static_assert(std::is_integral<T>::value, "Tensor::random: requires integral type");
    std::random_device rd;
    std::default_random_engine eng(rd());
    auto rand = std::bind(std::uniform_int_distribution<T>(low, high), eng);
    return build(std::move(shape), rand);
}

template <>
inline Tensor<float> Tensor<float>::random(Shape shape, float low, float high) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    auto rand = std::bind(std::uniform_real_distribution<float>(low, high), eng);
    return build(std::move(shape), rand);
}

template <>
inline Tensor<double> Tensor<double>::random(Shape shape, double low, double high) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    auto rand = std::bind(std::uniform_real_distribution<double>(low, high), eng);
    return build(std::move(shape), rand);
}

template <typename T>
Tensor<T>::Tensor(const Tensor& t) : Shaped(t)
{
    m_alloc_data = std::make_unique<T[]>(size());
    m_data = m_alloc_data.get();
    std::copy(t.begin(), t.end(), m_data);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& t) {
    if (size() != t.size() || m_alloc_data == nullptr) {
        m_alloc_data = std::make_unique<T[]>(t.size());
        m_data = m_alloc_data.get();
    }
    Shaped::operator=(t);
    std::copy(t.begin(), t.end(), m_data);
    return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor&& t) noexcept
    : Shaped(std::move(t)),
      m_data(std::exchange(t.m_data, nullptr)),
      m_alloc_data(std::move(t.m_alloc_data))
{
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& t) noexcept {
    Shaped::operator=(std::move(t));
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
    parallel_transform(begin(), end(), begin(), f);
    return *this;
}

template <typename T>
template <typename U, typename F>
inline Tensor<T>& Tensor<T>::apply(const Tensor<U>& y, F f) {
    assert(shape() == y.shape());
    parallel_transform(begin(), end(), y.begin(), begin(), f);
    return *this;
}

template <typename T>
template <typename F, typename U>
inline Tensor<U> Tensor<T>::transform(F f) const {
    Tensor<U> res(shape());
    parallel_transform(begin(), end(), res.begin(), f);
    return res;
}

template <typename T>
template <typename U, typename F>
inline void Tensor<T>::transformTo(Tensor<U>& target, F f) const {
    assert(shape() == target.shape());
    parallel_transform(begin(), end(), target.begin(), f);
}

template <typename T>
template <typename U, typename F, typename W>
inline Tensor<W> Tensor<T>::transform(const Tensor<U>& y, F f) const {
    assert(shape() == y.shape());
    Tensor<W> z(shape());
    parallel_transform(begin(), end(), y.begin(), z.begin(), f);
    return z;
}

template <typename T>
template <typename U, typename W, typename F>
inline void Tensor<T>::transformTo(Tensor<W>& z, const Tensor<U>& y, F f) const {
    assert(shape() == y.shape() && shape() == z.shape());
    parallel_transform(begin(), end(), y.begin(), z.begin(), f);
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
    parallel_transform(begin(), end(), y.begin(), y.begin(), f);
    return std::move(y);
}

template <typename T>
template <typename U>
inline Tensor<U> Tensor<T>::cast() const {
    return transform([](const T& x) { return static_cast<U>(x); });
}

//==-------------------------------------------------------------------------
// Tensor concrete transformations
//==-------------------------------------------------------------------------

template <typename T>
inline Tensor<T> abs(const Tensor<T>& x) {
    return x.transform([](T a) { return std::abs(a); });
}

template <typename T>
inline Tensor<T> abs(Tensor<T>&& x) {
    return std::move(x.apply([](T a) { return std::abs(a); }));
}

template <typename T>
inline void abs(const Tensor<T>& x, Tensor<T>& y) {
    assert(x.shape() == y.shape());
    x.transformTo(y, [](T a) { return std::abs(a);  });
}

template <typename T>
inline Tensor<T> neg(const Tensor<T>& x) {
    return x.transform([](T a) { return T(-a); });
}

template <typename T>
inline Tensor<T> neg(Tensor<T>&& x) {
    return std::move(x.apply([](T a) { return T(-a); }));
}

template <typename T>
inline void neg(const Tensor<T>& x, Tensor<T>& y) {
    assert(x.shape() == y.shape());
    x.transformTo(y, [](T a) { return T(-a);  });
}

template <typename T>
inline Tensor<T> sign(const Tensor<T>& x) {
    return x.transform([](T a) { return T((T(0)<a) - (a<T(0))); });
}

template <typename T>
inline Tensor<T> sign(Tensor<T>&& x) {
    return std::move(x.apply([](T a) { return T((T(0)<a) - (a<T(0))); }));
}

template <typename T>
inline void sign(const Tensor<T>& x, Tensor<T>& y) {
    assert(x.shape() == y.shape());
    x.transformTo(y, [](T a) { return T((T(0)<a) - (a<T(0))); });
}

template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
inline Tensor<T> reciprocal(const Tensor<T>& x) {
    return x.transform([](T a) { return T(1)/a; });
}

template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
inline Tensor<T> reciprocal(Tensor<T>&& x) {
    return std::move(x.apply([](T a) { return T(1)/a; }));
}

template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
inline void reciprocal(const Tensor<T>& x, Tensor<T>& y) {
    assert(x.shape() == y.shape());
    x.transformTo(y, [](T a) { return T(1)/a; });
}

#define DEFINE_TRANSFORM(name) \
template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>> \
inline Tensor<T> name(const Tensor<T>& x) { \
    return x.transform([](T a){ return std::name(a); }); \
} \
template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>> \
inline Tensor<T> name(Tensor<T>&& x) { \
    return std::move(x.apply([](T a){ return std::name(a); })); \
} \
template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>> \
inline void name(const Tensor<T>& x, Tensor<T>& y) { \
    assert(x.shape() == y.shape()); \
    x.transformTo(y, [](T a){ return std::name(a);  }); \
}

DEFINE_TRANSFORM(floor)
DEFINE_TRANSFORM(ceil)
DEFINE_TRANSFORM(round)
DEFINE_TRANSFORM(sqrt)
DEFINE_TRANSFORM(exp)
DEFINE_TRANSFORM(log)
DEFINE_TRANSFORM(sin)
DEFINE_TRANSFORM(cos)
DEFINE_TRANSFORM(tan)
DEFINE_TRANSFORM(asin)
DEFINE_TRANSFORM(acos)
DEFINE_TRANSFORM(atan)
DEFINE_TRANSFORM(sinh)
DEFINE_TRANSFORM(cosh)
DEFINE_TRANSFORM(tanh)
DEFINE_TRANSFORM(asinh)
DEFINE_TRANSFORM(acosh)
DEFINE_TRANSFORM(atanh)
DEFINE_TRANSFORM(erf)

#undef DEFINE_TRANSFORM

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
    return neg(x);
}

template <typename T>
inline Tensor<T> operator-(Tensor<T>&& x) {
    return neg(std::move(x));
}

//==-------------------------------------------------------------------------
// Tensor operations
//==-------------------------------------------------------------------------

namespace impl {

template <typename T>
inline std::enable_if_t<std::is_same<T,float>::value || std::is_same<T,double>::value, T>
dot(size_t n, const T* A, const T* B) {
    return cblas::dot(n, A, 1, B, 1);
}

template <typename T>
std::enable_if_t<!(std::is_same<T,float>::value || std::is_same<T,double>::value), T>
dot(size_t n, const T* A, const T* B) {
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
        std::plus<T>());
}

template <typename T>
inline std::enable_if_t<cblas::RequireBlasType<T>>
gemv(size_t m, size_t n, const T* A, const T* B, T* C, size_t lda) {
    cblas::gemv(cblas::Layout::RowMajor, cblas::Transpose::NoTrans, m, n, T(1), A, lda, B, 1, T(0), C, 1);
}

template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
gemv(size_t m, size_t n, const T* A, const T* B, T* C, size_t lda) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m, GRAINSIZE), [&](auto&& r) {
        auto px = A + r.begin() * lda;
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
inline std::enable_if_t<cblas::RequireBlasType<T>>
gemm(size_t m, size_t n, size_t k, const T* A, const T* B, T* C, size_t lda, size_t ldb, size_t ldc) {
    cblas::gemm(cblas::Layout::RowMajor, cblas::Transpose::NoTrans, cblas::Transpose::NoTrans,
                m, n, k, T(1), A, lda, B, ldb, T(0), C, ldc);
}

template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
gemm(size_t m, size_t n, size_t k, const T* A, const T* B, T* C, size_t lda, size_t ldb, size_t ldc) {
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [&](auto &&r) {
        for (size_t i = r.rows().begin(); i != r.rows().end(); i++) {
            for (size_t j = r.cols().begin(); j != r.cols().end(); j++) {
                T v{};
                for (size_t t = 0; t < k; t++)
                    v += A[i * lda + t] * B[t * ldb + j];
                C[i * ldc + j] = std::move(v);
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
        auto n = A.extent(0);
        assert(n == B.extent(0));
        assert(C->is_vector() && 1 == C->extent(0));
        *C->data() = impl::dot(n, A.data(), B.data());
    } else if (A.is_matrix() && B.is_vector()) {
        auto m = A.extent(0), n = A.extent(1);
        assert(n == B.extent(0));
        assert(C->is_vector() && m == C->extent(0));
        impl::gemv(m, n, A.data(), B.data(), C->data(), A.stride(0));
    } else if ((A.is_vector() || A.is_matrix()) && B.is_matrix()) {
        Shape A_shape, B_shape, C_shape;

        if (A.is_vector()) {
            assert(C->is_vector());
            A_shape = Shape({1, A.extent(0)});
            B_shape = B.shape();
            C_shape = Shape({1, C->extent(0)});
        } else {
            assert(C->is_matrix());
            A_shape = A.shape();
            B_shape = B.shape();
            C_shape = C->shape();
        }

        assert(A_shape.extent(1) == B_shape.extent(0));
        assert(C_shape.extent(0) == A_shape.extent(0));
        assert(C_shape.extent(1) == B_shape.extent(1));

        impl::gemm(C_shape.extent(0), C_shape.extent(1), A_shape.extent(1),
                   A.data(), B.data(), C->data(),
                   A_shape.stride(0), B_shape.stride(0), C_shape.stride(0));
    } else {
        assert(false);
    }
    return *C;
}

template <typename T>
Tensor<T> inner(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.is_vector() && B.is_vector()) {
        assert(A.shape() == B.shape());
        Tensor<T> C({1});
        inner(A, B, &C);
        return C;
    } else if (A.is_matrix() && B.is_vector()) {
        assert(A.extent(1) == B.extent(0));
        Tensor<T> C({A.extent(0)});
        return inner(A, B, &C);
        return C;
    } else if (A.is_vector() && B.is_matrix()) {
        assert(A.extent(0) == B.extent(0));
        Tensor<T> C({B.extent(1)});
        inner(A, B, &C);
        return C;
    } else if (A.is_matrix() && B.is_matrix()) {
        auto m = A.extent(0), k = A.extent(1);
        auto p = B.extent(0), n = B.extent(1);
        assert(k == p);
        Tensor<T> C({m, n});
        inner(A, B, &C);
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
        return Tensor<T>::identity(x.extent(0));
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
std::enable_if_t<!cblas::RequireBlasType<T>>
gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
     const T& beta, Tensor<T>* C,
     bool transA = false, bool transB = false)
{
    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);
    const auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    assert(k == p);
    assert(C->shape() == Shape({m, n}));

    if (alpha == T(0)) {
        *C *= beta;
        return;
    }

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [&](auto&& r) {
        size_t incX = transA ? lda : 1;
        size_t incY = transB ? 1 : ldb;
        for (size_t i = r.rows().begin(); i != r.rows().end(); i++) {
            T* pz = &C->data()[i * ldc + r.cols().begin()];
            for (size_t j = r.cols().begin(); j != r.cols().end(); j++) {
                const T* px = A.data() + (transA ? i : i*lda);
                const T* py = B.data() + (transB ? j*ldb : j);
                T v = beta == T(0) ? T(0) : *pz * beta;
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
std::enable_if_t<cblas::RequireBlasType<T>>
gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
     const T& beta, Tensor<T>* C,
     bool transA = false, bool transB = false)
{
    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    assert(k == p);
    assert(C->shape() == Shape({m, n}));

    cblas::gemm(cblas::Layout::RowMajor,
                transA ? cblas::Transpose::Trans : cblas::Transpose::NoTrans,
                transB ? cblas::Transpose::Trans : cblas::Transpose::NoTrans,
                m, n, k, alpha, A.data(), A.stride(0), B.data(), B.stride(0),
                beta, C->data(), C->stride(0));
}

template <typename T>
Tensor<T> gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
               const T& beta, const Tensor<T>& C,
               bool transA = false, bool transB = false)
{
    Tensor<T> R = C;
    gemm(alpha, A, B, beta, &R, transA, transB);
    return R;
}

namespace impl {

#if HAS_MKL
template <typename T>
constexpr bool RequireMKL = cblas::RequireBlasType<T>;
#else
template <typename T>
constexpr bool RequireMKL = false;
#endif

// Simple case: transpose with copy
template <typename T>
std::enable_if_t<!RequireMKL<T>>
copy_transpose(size_t r, size_t c, const T* src, T* dst) {
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
std::enable_if_t<!RequireMKL<T>>
square_transpose(size_t n, T* A) {
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
std::enable_if_t<!RequireMKL<T>>
inplace_transpose(size_t r, size_t c, T* A) {
    // naive implementation
    Tensor<T> t({r, c}, A, A+r*c);
    copy_transpose(r, c, t.data(), A);
}

#if HAS_MKL
template <typename T>
std::enable_if_t<RequireMKL<T>>
copy_transpose(size_t r, size_t c, const T* src, T* dst) {
    mkl::omatcopy('R', 'T', r, c, T(1), src, c, dst, r);
}

template <typename T>
std::enable_if_t<RequireMKL<T>>
square_transpose(size_t n, T* A) {
    mkl::imatcopy('R', 'T', n, n, T(1), A, n, n);
}

template <typename T>
std::enable_if_t<RequireMKL<T>>
inplace_transpose(size_t r, size_t c, T* A) {
    mkl::imatcopy('R', 'T', r, c, T(1), A, c, r);
}
#endif

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
    auto r = from.extent(0), c = from.extent(1);
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
    auto r = A.extent(0), c = A.extent(1);
    Tensor<T> B = Tensor<T>({c, r});
    impl::copy_transpose(r, c, A.data(), B.data());
    return B;
}

} // namespace dlf
