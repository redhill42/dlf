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
    using pointer                   = value_type*;
    using const_pointer             = const value_type*;
    using size_type                 = size_t;
    using difference_type           = ptrdiff_t;
    using iterator                  = value_type*;
    using const_iterator            = const value_type*;
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

    shaped_iterator<T> begin(Shape shape)
        { return shaped_iterator<T>(std::move(shape), data(), 0); }
    shaped_iterator<T> end(Shape shape)
        { return shaped_iterator<T>(std::move(shape), data(), shape.size()); }
    const_shaped_iterator<T> begin(Shape shape) const
        { return const_shaped_iterator<T>(std::move(shape), data(), 0); }
    const_shaped_iterator<T> end(Shape shape) const
        { return const_shaped_iterator<T>(std::move(shape), data(), shape.size()); }

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
     * Create a scalar.
     *
     * @param value the scalar value
     */
    static Tensor scalar(const T& value);

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
     * Casting element type.
     *
     * @tparam U the target element type
     * @return the Tensor with new element type.
     */
    template <typename U>
    Tensor<U> cast() const;

private: // Formatting
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        if (!t.empty())
            printRec(os, t.shape(), 0, t.data());
        return os;
    }

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
Tensor<T>::Tensor(Shape shape, std::initializer_list<T> il)
    : Shaped(std::move(shape))
{
    init();
    assert(size() == il.size());
    std::copy(il.begin(), il.end(), m_data);
}

template <typename T>
Tensor<T>::Tensor(Shape shape, std::unique_ptr<T[]> data)
    : Shaped(std::move(shape)), m_alloc_data(std::move(data))
{
    m_data = m_alloc_data.get();
}

template <typename T>
inline Tensor<T> Tensor<T>::wrap(Shape shape, T* data) {
    return Tensor(std::move(shape), data);
}

template <typename T>
inline Tensor<T> Tensor<T>::scalar(const T& value) {
    Tensor<T> ret({1});
    *ret.data() = value;
    return ret;
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
inline bool operator==(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return lhs.shape() == rhs.shape() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T>
inline bool operator!=(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return !(lhs == rhs);
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
        for (int i = 0; ; i++) {
            out << *data++;
            if (i == d-1)
                break;
            out << ',';
        }
    } else {
        // intermediate levels, recursive
        for (int i = 0; ; i++) {
            data = printRec(out, shape, level+1, data);
            if (i == d-1)
                break;
            out << ',';
        }
    }
    out << ']';

    return data;
}

//==-------------------------------------------------------------------------
// Tensor unary transformations
//==-------------------------------------------------------------------------

/**
 * Transform tensor A's elements to tensor B by applying the given function.
 * The two tensor must have the same shape.
 */
template <typename T, typename U, typename F>
inline Tensor<U>& transformTo(const Tensor<T>& A, Tensor<U>& B, F f) {
    assert(A.shape() == B.shape());
    par::transform(A.begin(), A.end(), B.begin(), f);
    return B;
}

/**
 * Transform a tensor to a new tensor by applying the given unary function
 * on tensor's elements.
 */
template <typename T, typename F, typename U = cxx::invoke_result_t<F,T>>
inline Tensor<U> transform(const Tensor<T>& A, F f) {
    Tensor<U> B(A.shape());
    transformTo(A, B, f);
    return B;
}

template <typename T, typename F, typename = std::enable_if_t<std::is_same<cxx::invoke_result_t<F,T>,T>::value>>
inline Tensor<T> transform(Tensor<T>&& A, F f) {
    return std::move(transformTo(A, A, f));
}

//==-------------------------------------------------------------------------
// Tensor binary transformations
//==-------------------------------------------------------------------------

template <typename T, typename U, typename W, typename F>
Tensor<W>& transformTo(const Tensor<T>& A, const Tensor<U>& B, Tensor<W>& C, F f) {
    Shape final_shape = Shape::broadcast(A, B);
    Shape sA = A.shape().broadcast(final_shape);
    Shape sB = B.shape().broadcast(final_shape);
    int   axis;

    if (C.shape() != final_shape) {
        throw shape_error("incompatible shape");
    }

    if (A.shape() == B.shape()) {
        par::transform(A.begin(), A.end(), B.begin(), C.begin(), f);
    } else if (A.size() == 1) {
        par::transform(B.begin(), B.end(), C.begin(), [x=*A.data(),f](auto& y){ return f(x, y); });
    } else if (B.size() == 1) {
        par::transform(A.begin(), A.end(), C.begin(), [y=*B.data(),f](auto& x){ return f(x, y); });
    } else if ((axis = Shape::axis(A.shape(), B.shape())) != -1) {
        transformChannel(A, B, C, axis, f);
    } else if (sA.is_contiguous()) {
        par::transform(A.begin(), A.end(), B.begin(sB), C.begin(), f);
    } else if (sB.is_contiguous()) {
        par::transform(A.begin(sA), A.end(sB), B.begin(), C.begin(), f);
    } else {
        par::transform(A.begin(sA), A.end(sB), B.begin(sB), C.begin(), f);
    }

    return C;
}

template <typename T, typename U, typename W, typename F>
void transformChannel(const Tensor<T>& A, const Tensor<U>& B, Tensor<W>& C, size_t axis, F fn) {
    assert(B.is_vector() || Shape::axis(A.shape(), B.shape()) == axis);
    assert(axis < A.rank());
    assert(A.extent(axis) == B.size());
    assert(C.shape() == A.shape());

    size_t m = 1;
    for (int i = 0; i <= axis; i++)
        m *= A.extent(i);
    size_t n = A.size() / m;

    tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 32, 0, n, 32), [&](auto r) {
        auto offset = r.rows().begin()*n + r.cols().begin();
        auto px = A.data() + offset;
        auto py = B.data();
        auto pz = C.data() + offset;
        for (int id = r.rows().begin(); id < r.rows().end(); id++) {
            auto y = py[id % B.size()];
            std::transform(px, px+r.cols().size(), pz, [=](auto x){ return fn(x, y); });
            px += n, pz += n;
        }
    });
}

/**
 * Transform two tensors to a new tensor by applying the given binary function.
 */
template <typename T, typename U, typename F, typename W = cxx::invoke_result_t<F,T,U>>
inline Tensor<W> transform(const Tensor<T>& A, const Tensor<U>& B, F f) {
    Tensor<W> C(Shape::broadcast(A, B));
    transformTo(A, B, C, f);
    return C;
}

template <typename T, typename F, typename = std::enable_if_t<std::is_same<cxx::invoke_result_t<F,T,T>,T>::value>>
inline Tensor<T> transform(Tensor<T>&& A, const Tensor<T>& B, F f) {
    if (A.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, A, f));
    else
        return transform(A, B, f);
}

template <typename T, typename F, typename = std::enable_if_t<std::is_same<cxx::invoke_result_t<F,T,T>,T>::value>>
inline Tensor<T> transform(const Tensor<T>& A, Tensor<T>&& B, F f) {
    if (B.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, B, f));
    else
        return transform(A, B, f);
}

template <typename T, typename F, typename = std::enable_if_t<std::is_same<cxx::invoke_result_t<F,T,T>,T>::value>>
inline Tensor<T> transform(Tensor<T>&& A, Tensor<T>&& B, F f) {
    Shape final_shape = Shape::broadcast(A, B);
    if (final_shape == A.shape())
        return std::move(transformTo(A, B, A, f));
    else if (final_shape == B.shape())
        return std::move(transformTo(A, B, B, f));
    else
        return transform(A, B, f);
}

template <typename T>
template <typename F>
inline Tensor<T>& Tensor<T>::apply(F f) {
    return transformTo(*this, *this, f);
}

template <typename T>
template <typename U, typename F>
inline Tensor<T>& Tensor<T>::apply(const Tensor<U>& y, F f) {
    return transformTo(*this, y, *this, f);
}

template <typename T>
template <typename U>
inline Tensor<U> Tensor<T>::cast() const {
    return transform(*this, [](const T& x) { return static_cast<U>(x); });
}

template <typename T>
void copy(const Tensor<T>& src, const Shape& shape, Tensor<T>& dst) {
    assert(dst.shape() == shape);
    if (src.data() == dst.data())
        return;
    if (src.size() == 1) {
        std::fill(dst.begin(), dst.end(), *src.data());
    } if (src.size() == shape.size() && shape.is_contiguous()) {
        par::copy(src.begin(), src.end(), dst.begin());
    } else {
        par::copy(src.begin(shape), src.end(shape), dst.begin());
    }
}

template <typename T>
inline void flat_copy(const Tensor<T>& src, Tensor<T>& dst) {
    assert(src.size() == dst.size());
    if (src.data() != dst.data()) {
        par::copy(src.begin(), src.end(), dst.begin());
    }
}

template <typename T>
inline void broadcast(const Tensor<T>& src, Tensor<T>& dst) {
    copy(src, src.shape().broadcast(dst.shape()), dst);
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
 * Perform dot product on two tensors. The tensors must be vector
 * or matrix and have compatible dimensions.
 */
template <typename T>
Tensor<T>& dot(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>* C) {
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
Tensor<T> dot(const Tensor<T>& A, const Tensor<T>& B) {
    if (A.is_vector() && B.is_vector()) {
        assert(A.shape() == B.shape());
        Tensor<T> C({1});
        dot(A, B, &C);
        return C;
    } else if (A.is_matrix() && B.is_vector()) {
        assert(A.extent(1) == B.extent(0));
        Tensor<T> C({A.extent(0)});
        dot(A, B, &C);
        return C;
    } else if (A.is_vector() && B.is_matrix()) {
        assert(A.extent(0) == B.extent(0));
        Tensor<T> C({B.extent(1)});
        dot(A, B, &C);
        return C;
    } else if (A.is_matrix() && B.is_matrix()) {
        auto m = A.extent(0), k = A.extent(1);
        auto p = B.extent(0), n = B.extent(1);
        assert(k == p);
        Tensor<T> C({m, n});
        dot(A, B, &C);
        return C;
    } else {
        assert(false);
        return {};
    }
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
        *C *= Tensor<T>::scalar(beta);
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
void gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
          const T& beta, const Tensor<T>& C, Tensor<T>& Y,
          bool transA = false, bool transB = false)
{
    broadcast(C, Y);
    gemm(alpha, A, B, beta, &Y, transA, transB);
}

template <typename T>
Tensor<T> gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
               const T& beta, const Tensor<T>& C,
               bool transA = false, bool transB = false)
{
    assert(A.is_matrix() && B.is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);
    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    assert(k == p);

    auto Y = broadcast(C, {m, n});
    gemm(alpha, A, B, beta, &Y, transA, transB);
    return Y;
}

/**
 * The outer product on tensors is typically referred to as the tensor product.
 * Given a tensor a of order q with dimensions (i1, ..., iq), and a tensor b
 * of order r with dimensions (j1, ..., jr), their outer product c is of order
 * q + r with dimensions (k1, ..., kq+r) which are the i dimensions followed
 * by the j dimensions.
 */
template <typename T, typename U, typename F, typename W = cxx::invoke_result_t<F,T,U>>
Tensor<W> outer(const Tensor<T>& A, const Tensor<U>& B, F f) {
    std::vector<size_t> dimA, dimB, dimC;
    for (size_t i = 0; i < A.rank(); i++) {
        dimA.push_back(A.extent(i));
        dimB.push_back(1);
        dimC.push_back(A.extent(i));
    }
    for (size_t i = 0; i < B.rank(); i++) {
        dimA.push_back(1);
        dimB.push_back(B.extent(i));
        dimC.push_back(B.extent(i));
    }

    auto sC = Shape(dimC);
    auto sA = Shape(dimA).broadcast(sC);
    auto sB = Shape(dimB).broadcast(sC);

    Tensor<W> C(sC);
    par::transform(A.begin(sA), A.end(sA), B.begin(sB), C.begin(), f);
    return C;
}

template <typename T, typename U, typename W = std::common_type_t<T,U>>
inline Tensor<W> outer(const Tensor<T>& A, const Tensor<U>& B) {
    return outer(A, B, [](const T& a, const U& b) -> W { return a * b; });
}

//==-------------------------------------------------------------------------
// Tensor DNN operations
//==-------------------------------------------------------------------------

namespace dnn {
template <typename T>
void batch_norm(const Tensor<T>& X, Tensor<T>& Y,
                const Tensor<T>& scale, const Tensor<T>& bias,
                const Tensor<T>& mean, const Tensor<T>& var,
                const T epsilon = T(1e-5))
{
    assert(X.shape() == Y.shape());
    auto batches  = X.extent(0);
    auto channels = X.extent(1);
    auto spatial  = X.size() / (batches * channels);

    assert(scale.is_vector() && scale.extent(0) == channels);
    assert(bias.is_vector() && bias.extent(0) == channels);
    assert(mean.is_vector() && mean.extent(0) == channels);
    assert(var.is_vector() && var.extent(0) == channels);

    const T* x = X.data();
          T* y = Y.data();
    const T* s = scale.data();
    const T* b = bias.data();
    const T* m = mean.data();

    T* v = reinterpret_cast<T*>(alloca(channels * sizeof(T)));
    std::transform(var.begin(), var.end(), v, [=](auto x) {
        return std::sqrt(x + epsilon);
    });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, spatial, 256), [=](auto& r) {
        for (size_t bat = 0; bat < batches; bat++) {
            for (size_t c = 0; c < channels; c++) {
                auto offset = (bat * channels + c) * spatial + r.begin();
                auto px = x + offset;
                auto py = y + offset;
                for (auto n = r.size(); n--; ) {
                    *py++ = s[c] * (*px++ - m[c]) / v[c] + b[c];
                }
            }
        }
    });
}

namespace detail {
template <typename T>
void im2col(const size_t channels,
            const size_t input_h, const size_t input_w,
            const size_t output_h, const size_t output_w,
            const size_t kernel_h, const size_t kernel_w,
            const size_t pad_h, const size_t pad_w,
            const size_t stride_h, const size_t stride_w,
            const size_t dilation_h, const size_t dilation_w,
            const T* im_buffer, T* col_buffer)
{
    tbb::parallel_for(tbb::blocked_range2d<int>(0, output_w, 32, 0, output_h*channels, 32), [&](auto& r) {
        for (int w_id = r.rows().begin(); w_id < r.rows().end(); w_id++) {
            for (int hc_id = r.cols().begin(); hc_id < r.cols().end(); hc_id++) {
                int c_id = hc_id / output_h;
                int h_id = hc_id - c_id * output_h;

                for (int kh_id = 0; kh_id < kernel_h; kh_id++) {
                    for (int kw_id = 0; kw_id < kernel_w; kw_id++) {
                        // Retrieves the input value
                        int h_index = kh_id * dilation_h + stride_h * h_id - pad_h;
                        int w_index = kw_id * dilation_w + stride_w * w_id - pad_w;
                        T val{};
                        if (h_index >= 0 && h_index < input_h &&
                            w_index >= 0 && w_index < input_w) {
                            int input_index = (c_id * input_h + h_index) * input_w + w_index;
                            val = im_buffer[input_index];
                        }

                        // Sets the output value
                        int kernel_index = kernel_h * kernel_w - kw_id - kernel_w * kh_id - 1;
                        int output_index = c_id * output_w * output_h * kernel_h * kernel_w +
                                           kernel_index * output_w * output_h +
                                           h_id * output_w + w_id;
                        col_buffer[output_index] = val;
                    }
                }
            }
        }
    });
}

template <typename T>
void maxpool(const size_t batches, const size_t channels,
             const size_t input_h, const size_t input_w,
             const size_t output_h, const size_t output_w,
             const size_t kernel_h, const size_t kernel_w,
             const size_t pad_h, const size_t pad_w,
             const size_t stride_h, const size_t stride_w,
             const size_t dilation_h, const size_t dilation_w,
             const T* input, T* output)
{
    tbb::parallel_for(tbb::blocked_range3d<int>(0, batches, 1, 0, output_w, 32, 0, output_h*channels, 32), [&](auto& r) {
        for (int b_id = r.pages().begin(); b_id < r.pages().end(); b_id++) {
            for (int w_id = r.rows().begin(); w_id < r.rows().end(); w_id++) {
                for (int hc_id = r.cols().begin(); hc_id < r.cols().end(); hc_id++) {
                    int c_id = hc_id / output_h;
                    int h_id = hc_id - c_id * output_h;

                    T v = std::numeric_limits<T>::lowest();
                    for (int kh_id = 0; kh_id < kernel_h; kh_id++) {
                        int h_index = kh_id * dilation_h + stride_h * h_id - pad_h;
                        if (h_index >= 0 && h_index < input_h) {
                            for (int kw_id = 0; kw_id < kernel_w; kw_id++) {
                                int w_index = kw_id * dilation_w + stride_w * w_id - pad_w;
                                if (w_index >= 0 && w_index < input_w) {
                                    int input_index =
                                        ((b_id * channels + c_id) * input_h + h_index) * input_w + w_index;
                                    v = std::max(v, input[input_index]);
                                }
                            }
                        }
                    }

                    int output_index = ((b_id * channels + c_id) * output_h + h_id) * output_w + w_id;
                    output[output_index] = v;
                }
            }
        }
    });
}

template <typename T>
void avgpool(const size_t batches, const size_t channels,
             const size_t input_h, const size_t input_w,
             const size_t output_h, const size_t output_w,
             const size_t kernel_h, const size_t kernel_w,
             const size_t pad_h, const size_t pad_w,
             const size_t stride_h, const size_t stride_w,
             const size_t dilation_h, const size_t dilation_w,
             const bool count_include_pad,
             const T* input, T* output)
{
    tbb::parallel_for(tbb::blocked_range3d<int>(0, batches, 1, 0, output_w, 32, 0, output_h*channels, 32), [&](auto& r) {
        for (int b_id = r.pages().begin(); b_id < r.pages().end(); b_id++) {
            for (int w_id = r.rows().begin(); w_id < r.rows().end(); w_id++) {
                for (int hc_id = r.cols().begin(); hc_id < r.cols().end(); hc_id++) {
                    int c_id = hc_id / output_h;
                    int h_id = hc_id - c_id * output_h;

                    T sum = 0; int count = 0;
                    for (int kh_id = 0; kh_id < kernel_h; kh_id++) {
                        int h_index = kh_id * dilation_h + stride_h * h_id - pad_h;
                        if (h_index >= 0 && h_index < input_h) {
                            for (int kw_id = 0; kw_id < kernel_w; kw_id++) {
                                int w_index = kw_id * dilation_w + stride_w * w_id - pad_w;
                                if (w_index >= 0 && w_index < input_w) {
                                    int input_index =
                                        ((b_id * channels + c_id) * input_h + h_index) * input_w + w_index;
                                    sum += input[input_index];
                                    count++;
                                }
                            }
                        }
                    }

                    T val = count_include_pad ? sum/(kernel_h*kernel_w) : sum/count;
                    int output_index = ((b_id * channels + c_id) * output_h + h_id) * output_w + w_id;
                    output[output_index] = val;
                }
            }
        }
    });
}
} // namespace detail

template <typename T>
void conv2d(const Tensor<T>& X, const Tensor<T>& W, Tensor<T>& Y, const FilterShape2D& filter) {
    assert(X.shape() == filter.input_shape());
    assert(W.shape() == filter.kernel_shape());
    assert(Y.shape() == filter.output_shape());

    const auto m = filter.num_kernels();
    const auto k = filter.channels() * filter.kernel_h() * filter.kernel_w();
    const auto n = filter.output_h() * filter.output_w();
    Tensor<T> work({k, n});

    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    for (size_t b = 0; b < filter.batches(); b++) {
        detail::im2col(filter.channels(), filter.height(), filter.width(),
                       filter.output_h(), filter.output_w(),
                       filter.kernel_h(), filter.kernel_w(),
                       filter.pad_h(), filter.pad_w(),
                       filter.stride_h(), filter.stride_w(),
                       filter.dilation_h(), filter.dilation_w(),
                       x_buffer, work.data());

        cblas::gemm(cblas::Layout::RowMajor,
                    cblas::Transpose::NoTrans, cblas::Transpose::NoTrans,
                    m, n, k, T{1},
                    W.data(), W.stride(0),
                    work.data(), work.stride(0), T{0},
                    y_buffer, Y.stride(1));

        x_buffer += X.stride(0);
        y_buffer += Y.stride(0);
    }
}

template <typename T>
void maxpool(const Tensor<T>& X, Tensor<T>& Y, const FilterShape2D& filter) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    detail::maxpool(filter.batches(), filter.channels(),
                    filter.height(), filter.width(),
                    filter.output_h(), filter.output_w(),
                    filter.kernel_h(), filter.kernel_w(),
                    filter.pad_h(), filter.pad_w(),
                    filter.stride_h(), filter.stride_w(),
                    filter.dilation_h(), filter.dilation_w(),
                    X.data(), Y.data());
}

template <typename T>
void avgpool(const Tensor<T>& X, Tensor<T>& Y, const FilterShape2D& filter, bool count_include_pad) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    detail::avgpool(filter.batches(), filter.channels(),
                    filter.height(), filter.width(),
                    filter.output_h(), filter.output_w(),
                    filter.kernel_h(), filter.kernel_w(),
                    filter.pad_h(), filter.pad_w(),
                    filter.stride_h(), filter.stride_w(),
                    filter.dilation_h(), filter.dilation_w(),
                    count_include_pad,
                    X.data(), Y.data());
}

template <typename T>
void global_maxpool(const Tensor<T>& X, Tensor<T>& Y) {
    assert(X.rank() >= 3);
    assert(X.rank() == Y.rank());
    auto M = X.extent(0) * X.extent(1);
    auto N = X.size() / M;
    assert(Y.size() == M);

    size_t grainsize = std::max(size_t(1), GRAINSIZE / N);
    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, M, grainsize), [=](const auto& r) {
        for (int b = r.begin(); b < r.end(); b++) {
            y_buffer[b] = tbb::parallel_reduce(
                tbb::blocked_range<int>(0, N, GRAINSIZE),
                std::numeric_limits<T>::lowest(),
                [=](const auto& r, T acc) {
                    auto px = x_buffer + b*N + r.begin();
                    for (size_t k = r.size(); k-- != 0; )
                        acc = std::max(acc, *px++);
                    return acc;
                },
                [](auto x, auto y) { return std::max(x, y); });
        }
    });
}

template <typename T>
void global_avgpool(const Tensor<T>& X, Tensor<T>& Y) {
    assert(X.rank() >= 3);
    assert(X.rank() == Y.rank());
    auto M = X.extent(0) * X.extent(1);
    auto N = X.size() / M;
    assert(Y.size() == M);

    size_t grainsize = std::max(size_t(1), GRAINSIZE / N);
    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, M, grainsize), [=](const auto& r) {
        for (int b = r.begin(); b < r.end(); b++) {
            auto val = tbb::parallel_reduce(
                tbb::blocked_range<int>(0, N, grainsize),
                T{},
                [=](const auto& r, T acc) {
                    auto px = x_buffer + b*N + r.begin();
                    for (size_t k = r.size(); k-- != 0; )
                        acc += *px++;
                    return acc;
                },
                std::plus<>());
            y_buffer[b] = val / N;
        }
    });
}

template <typename T>
void softmax(const Tensor<T>& X, Tensor<T>& Y, int axis = 1) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("softmax: invalid axis");

    auto dims = X.shape().extents();
    auto M = std::accumulate(dims.begin(), dims.begin()+axis, size_t(1), std::multiplies<>());
    auto N = X.size() / M;

    assert(Y.shape() == X.shape() || Y.shape() == Shape({M, N}));

    size_t grainsize = std::max(size_t(1), GRAINSIZE / N);
    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, M, grainsize), [=](const auto& r) {
        for (int b = r.begin(); b < r.end(); b++) {
            auto px = x_buffer + b*N;
            auto py = y_buffer + b*N;

            T amax = px[0];
            for (size_t i = 1; i < N; i++) {
                amax = std::max(amax, px[i]);
            }

            T asum = 0;
            for (size_t i = 0; i < N; i++) {
                py[i] = std::exp(px[i] - amax);
                asum += py[i];
            }
            for (size_t i = 0; i < N; i++) {
                py[i] /= asum;
            }
        }
    });
}

template <typename T>
Tensor<T> softmax(const Tensor<T>& X, int axis = 1, bool keepdims = true) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("softmax: invalid axis");

    if (keepdims) {
        Tensor<T> Y(X.shape());
        softmax(X, Y, axis);
        return Y;
    } else {
        auto dims = X.shape().extents();
        auto M = std::accumulate(dims.begin(), dims.begin()+axis, size_t(1), std::multiplies<>());
        auto N = X.size() / M;

        Tensor<T> Y({M, N});
        softmax(X, Y, axis);
        return Y;
    }
}

template <typename T>
Tensor<T> softmax(Tensor<T>&& X, int axis = 1) {
    softmax(X, X, axis);
    return std::move(X);
}

} // namespace dnn
} // namespace dlf
