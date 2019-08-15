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

template <typename T> class TensorView;

/**
 * Tensor is a geometric object that maps in a multi-linear manner geometric
 * vectors, scalars, and other tensors to a resulting tensor.
 *
 * @tparam T the data type of the tensor.
 */
template <typename T>
class Tensor : public Shaped {
    T* m_data = nullptr;
    std::shared_ptr<T> m_alloc_data;

    void init() {
        m_alloc_data = std::shared_ptr<T>(new T[size()](), std::default_delete<T[]>());
        m_data = m_alloc_data.get();
    }

    friend class TensorView<T>;

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
     * Construct a tensor with given dimension and preallocated data. It's the
     * caller's responsibility to allocate enough memory space to store the
     * tensor data, and encapsulate the data into a shared_ptr. The memory space
     * allocated by caller will be freed when this tensor is no longer used.
     *
     * @param shape the tensor dimensions.
     * @param data the preallocated tensor data.
     */
    Tensor(Shape shape, std::shared_ptr<T> data);

    /**
     * Construct a tensor from a tensor view. The contents of the tensor view is
     * copied into newly created tensor and the shape is normalized.
     */
    explicit Tensor(const TensorView<T>& view);

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
     * Create a tensor that fill with constant value.
     *
     * @param shape the tensor dimension
     * @param value the constant value
     */
    static Tensor fill(Shape shape, const T& value);

    /**
     * Fill the tensor with a scalar value.
     *
     * @param value the constant scalar value.
     */
    Tensor& fill(const T& value);

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

    Tensor& operator=(const TensorView<T>& v);

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
     * Returns a slice given by the index.
     */
    Tensor<T> operator[](int index);

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
        os << t.shape();
        if (!t.empty())
            printRec(os, t.shape(), 0, t.data());
        return os;
    }

    static const T* printRec(std::ostream& out, const Shape& shape, size_t level, const T* data);
};

template <typename T>
class TensorView : public Shaped {
    T* m_data;
    std::shared_ptr<T> m_alloc_data;

public:
    TensorView(Shape shape, const Tensor<T>& src);
    TensorView(Shape shape, const TensorView<T>& src);

public: // Container View
    using value_type                = T;
    using reference                 = value_type&;
    using const_reference           = const value_type&;
    using pointer                   = value_type*;
    using const_pointer             = const value_type*;
    using size_type                 = size_t;
    using difference_type           = ptrdiff_t;
    using iterator                  = shaped_iterator<T>;
    using const_iterator            = const_shaped_iterator<T>;
    using reverse_iterator          = std::reverse_iterator<iterator>;
    using const_reverse_iterator    = std::reverse_iterator<const_iterator>;

    iterator begin() { return iterator(shape(), data(), 0); }
    const_iterator begin() const { return const_iterator(shape(), data(), 0); }
    iterator end() { return iterator(shape(), data(), size()); }
    const_iterator end() const { return const_iterator(shape(), data(), size()); }

    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    const_reverse_iterator crbegin() const { return rbegin(); }
    const_reverse_iterator crend() const { return rend(); }

public: // Attributes
    T* data() noexcept { return m_data; }
    const T* data() const noexcept { return m_data; }

    template <typename... Args, typename = std::enable_if_t<cxx::conjunction<std::is_convertible<Args, size_t>...>::value>>
    const T& operator()(Args... args) const noexcept;

    template <typename... Args, typename = std::enable_if_t<cxx::conjunction<std::is_convertible<Args, size_t>...>::value>>
    T& operator()(Args... args) noexcept;

    operator Tensor<T>() const { return Tensor<T>(*this); }

public: // Operations
    template <typename F>
    TensorView& apply(F f);

    template <typename U>
    Tensor<U> cast() const;

    TensorView& fill(const T& value) {
        std::fill(begin(), end(), value);
        return *this;
    }

private: // Formatting
    friend std::ostream& operator<<(std::ostream& os, const TensorView& v) {
        os << v.shape();
        if (!v.empty())
            printRec(os, v.shape(), 0, v.begin());
        return os;
    }

    static const_iterator printRec(std::ostream& out, const Shape& shape, size_t level, const_iterator cur);
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
Tensor<T>::Tensor(Shape shape, std::shared_ptr<T> data)
    : Shaped(std::move(shape)), m_alloc_data(std::move(data))
{
    m_data = m_alloc_data.get();
}

template <typename T>
Tensor<T>::Tensor(const TensorView<T>& view) : Shaped(view.shape()) {
    init();
    std::copy(view.begin(), view.end(), m_data);
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
Tensor<T> Tensor<T>::fill(Shape shape, const T& value) {
    Tensor<T> res(std::move(shape));
    std::fill(res.begin(), res.end(), value);
    return res;
}

template <typename T>
inline Tensor<T>& Tensor<T>::fill(const T& value) {
    std::fill(begin(), end(), value);
    return *this;
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
Tensor<T>::Tensor(const Tensor& t) : Shaped(t) {
    init();
    std::copy(t.begin(), t.end(), m_data);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& t) {
    auto old_size = size();
    Shaped::operator=(t);
    if (size() != old_size || m_alloc_data == nullptr)
        init();
    std::copy(t.begin(), t.end(), m_data);
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const TensorView<T>& v) {
    auto old_size = size();
    Shaped::operator=(v);
    if (size() != old_size || m_alloc_data == nullptr)
        init();
    std::copy(v.begin(), v.end(), m_data);
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
inline Tensor<T> Tensor<T>::operator[](int index) {
    auto slice_shape = shape().slice({{index, index+1}});
    return wrap(slice_shape, data() + slice_shape.offset());
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
// TensorView implementation
//==-------------------------------------------------------------------------

template <typename T>
TensorView<T>::TensorView(Shape shape, const Tensor<T>& src)
    : Shaped(std::move(shape), true),
      m_data(src.m_data),
      m_alloc_data(src.m_alloc_data)
{}

template <typename T>
TensorView<T>::TensorView(Shape shape, const TensorView<T>& src)
    : Shaped(std::move(shape), true),
      m_data(src.m_data),
      m_alloc_data(src.m_alloc_data)
{}

template <typename T>
inline bool operator==(const TensorView<T>& lhs, const TensorView<T>& rhs) {
    return lhs.shape() == rhs.shape() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T>
inline bool operator==(const Tensor<T>& lhs, const TensorView<T>& rhs) {
    return lhs.shape() == rhs.shape() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T>
inline bool operator==(const TensorView<T>& lhs, const Tensor<T>& rhs) {
    return lhs.shape() == rhs.shape() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T>
inline bool operator!=(const TensorView<T>& lhs, const TensorView<T>& rhs) {
    return !(lhs == rhs);
}

template <typename T>
inline bool operator!=(const Tensor<T>& lhs, const TensorView<T>& rhs) {
    return !(lhs == rhs);
}

template <typename T>
inline bool operator!=(const TensorView<T>& lhs, const Tensor<T>& rhs) {
    return !(lhs == rhs);
}

template <typename T>
template <typename... Args, typename>
inline const T& TensorView<T>::operator()(Args... args) const noexcept {
    return data()[shape().offset({size_t(args)...})];
}

template <typename T>
template <typename... Args, typename>
inline T& TensorView<T>::operator()(Args... args) noexcept {
    return data()[shape().offset({size_t(args)...})];
}

template <typename T>
typename TensorView<T>::const_iterator TensorView<T>::printRec(
    std::ostream& out, const Shape& shape, size_t level, const_iterator cur)
{
    auto d = shape.extent(level);

    out << '[';
    if (level == shape.rank()-1) {
        // last level, printing data
        for (int i = 0; ; i++) {
            out << *cur++;
            if (i == d-1)
                break;
            out << ',';
        }
    } else {
        // intermediate levels, recursive
        for (int i = 0; ; i++) {
            cur = printRec(out, shape, level+1, cur);
            if (i == d-1)
                break;
            out << ',';
        }
    }
    out << ']';

    return cur;
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

template <typename T, typename U, typename F>
inline TensorView<U>& transformTo(const Tensor<T>& A, TensorView<U>& B, F f) {
    assert(A.shape() == B.shape());
    par::transform(A.begin(), A.end(), B.begin(), f);
    return B;
}

template <typename T, typename U, typename F>
inline Tensor<U>& transformTo(const TensorView<T>& A, Tensor<U>& B, F f) {
    assert(A.shape() == B.shape());
    par::transform(A.begin(), A.end(), B.begin(), f);
    return B;
}

template <typename T, typename U, typename F>
inline TensorView<U>& transformTo(const TensorView<T>& A, Tensor<U>& B, F f) {
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

template <typename T, typename F, typename U = cxx::invoke_result_t<F,T>>
inline Tensor<U> transform(const TensorView<T>& A, F f) {
    Tensor<U> B(A.shape());
    transformTo(A, B, f);
    return B;
}

template <typename T>
template <typename F>
inline Tensor<T>& Tensor<T>::apply(F f) {
    return transformTo(*this, *this, f);
}

template <typename T>
template <typename F>
inline TensorView<T>& TensorView<T>::apply(F f) {
    return transformTo(*this, *this, f);
}

template <typename T>
template <typename U>
inline Tensor<U> Tensor<T>::cast() const {
    return transform(*this, [](const T& x) { return static_cast<U>(x); });
}

template <typename T>
template <typename U>
inline Tensor<U> TensorView<T>::cast() const {
    return transform(*this, [](const T& x) { return static_cast<U>(x); });
}

namespace impl {
template <typename T>
void reorder(const Shape& src_shape, const T* src_data, const size_t src_size,
             const Shape& dst_shape, T* dst_data)
{
    assert(src_shape == dst_shape);

    if (dst_shape.is_contiguous()) {
        if (src_size == 1) {
            std::fill(dst_data + dst_shape.offset(),
                      dst_data + dst_shape.offset() + dst_shape.size(),
                      src_data[src_shape.offset()]);
            return;
        }

        if (src_shape.is_contiguous()) {
            if (src_data != dst_data || src_shape.offset() != dst_shape.offset()) {
                par::copy(src_data + src_shape.offset(),
                          src_data + src_shape.offset() + src_shape.size(),
                          dst_data + dst_shape.offset());
            }
            return;
        }

        par::copy(const_shaped_iterator<T>(src_shape, src_data, 0),
                  const_shaped_iterator<T>(src_shape, src_data, src_shape.size()),
                  dst_data + dst_shape.offset());
    } else {
        if (src_size == 1) {
            std::fill(shaped_iterator<T>(dst_shape, dst_data, 0),
                      shaped_iterator<T>(dst_shape, dst_data, dst_shape.size()),
                      src_data[src_shape.offset()]);
            return;
        }

        if (src_shape.is_contiguous()) {
            par::copy(src_data + src_shape.offset(),
                      src_data + src_shape.offset() + src_shape.size(),
                      shaped_iterator<T>(dst_shape, dst_data, 0));
            return;
        }

        par::copy(const_shaped_iterator<T>(src_shape, src_data, 0),
                  const_shaped_iterator<T>(src_shape, src_data, src_shape.size()),
                  shaped_iterator<T>(dst_shape, dst_data, 0));
    }
}
}

template <typename T>
inline void reorder(const Tensor<T>& src, const Shape& src_shape, Tensor<T>& dst, const Shape& dst_shape) {
    impl::reorder(src_shape, src.data(), src.size(), dst_shape, dst.data());
}

template <typename T>
inline void reorder(const Tensor<T>& src, const Shape& src_shape, Tensor<T>& dst) {
    reorder(src, src_shape, dst, dst.shape());
}

template <typename T>
inline void reorder(const TensorView<T>& src, TensorView<T>& dst) {
    impl::reorder(src.shape(), src.data(), src.size(), dst.shape(), dst.data());
}

template <typename T>
inline void reorder(const TensorView<T>& src, TensorView<T>&& dst) {
    impl::reorder(src.shape(), src.data(), src.size(), dst.shape(), dst.data());
}

template <typename T>
inline void reorder(const TensorView<T>& src, Tensor<T>& dst) {
    impl::reorder(src.shape(), src.data(), src.size(), dst.shape(), dst.data());
}

template <typename T>
inline void reorder(const Tensor<T>& src, TensorView<T>& dst) {
    impl::reorder(src.shape(), src.data(), src.size(), dst.shape(), dst.data());
}

template <typename T>
inline void reorder(const Tensor<T>& src, TensorView<T>&& dst) {
    impl::reorder(src.shape(), src.data(), src.size(), dst.shape(), dst.data());
}

template <typename T>
inline void flat_copy(const Tensor<T>& src, Tensor<T>& dst) {
    assert(src.size() == dst.size());
    if (src.data() != dst.data()) {
        par::copy(src.begin(), src.end(), dst.begin());
    }
}

//==-------------------------------------------------------------------------
// Tensor binary transformations
//==-------------------------------------------------------------------------

namespace impl {
template <typename T, typename U, typename IC, typename F>
void transformChannel(const Shape& shape_A, const T* data_A,
                      const Shape& shape_B, const U* data_B,
                      const Shape& shape_C, IC begin_C,
                      int axis, F f)
{
    assert(shape_B.rank() == 1 || shape_A.find_channel_axis(shape_B) == axis);
    assert(axis < shape_A.rank());
    assert(shape_A.extent(axis) == shape_B.size());
    assert(shape_C == shape_A);

    size_t m = 1;
    for (int i = 0; i <= axis; i++)
        m *= shape_A.extent(i);
    size_t n = shape_A.size() / m;

    tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 32, 0, n, 32), [&](auto r) {
        auto offset = r.rows().begin()*n + r.cols().begin();
        auto px = data_A + shape_A.offset() + offset;
        auto py = data_B + shape_B.offset();
        auto pz = begin_C + offset;
        for (int id = r.rows().begin(); id < r.rows().end(); id++) {
            auto y = py[id % shape_B.size()];
            std::transform(px, px+r.cols().size(), pz, [=](auto x){ return f(x, y); });
            px += n, pz += n;
        }
    });
}

template <typename T, typename U, typename F, typename IteratorC>
void transformTo(const Shape& shape_A, const T* data_A, const size_t size_A,
                 const Shape& shape_B, const U* data_B, const size_t size_B,
                 F f, const Shape& shape_C, IteratorC begin_C)
{
    Shape final_shape = Shape::broadcast(shape_A, shape_B);
    Shape sA = shape_A.broadcast(final_shape);
    Shape sB = shape_B.broadcast(final_shape);
    int   axis;

    if (shape_C != final_shape) {
        throw shape_error("incompatible shape");
    }

    if (shape_A.is_contiguous() && shape_B.is_contiguous()) {
        if (shape_A == shape_B) {
            assert(shape_A == sA && shape_B == sB);
            par::transform(data_A + sA.offset(), data_A + sA.offset() + sA.size(),
                           data_B + sB.offset(),
                           begin_C, f);
        } else if (size_A == 1) {
            par::transform(data_B + sB.offset(), data_B + sB.offset() + sB.size(),
                           begin_C,
                           [x = data_A[shape_A.offset()], f](auto& y) { return f(x, y); });
        } else if (size_B == 1) {
            par::transform(data_A + sA.offset(), data_A + sA.offset() + sA.size(),
                           begin_C,
                           [y = data_B[shape_B.offset()], f](auto& x) { return f(x, y); });
        } else if ((axis = shape_A.find_channel_axis(shape_B)) != -1) {
            transformChannel(shape_A, data_A, shape_B, data_B, shape_C, begin_C, axis, f);
        } else {
            par::transform(const_shaped_iterator<T>(sA, data_A, 0),
                           const_shaped_iterator<T>(sA, data_A, sA.size()),
                           const_shaped_iterator<U>(sB, data_B, 0),
                           begin_C, f);
        }
    } else {
        if (sA.is_contiguous()) {
            par::transform(data_A + sA.offset(), data_A + sA.offset() + sA.size(),
                           const_shaped_iterator<U>(sB, data_B, 0),
                           begin_C, f);
        } else if (sB.is_contiguous()) {
            par::transform(const_shaped_iterator<T>(sA, data_A, 0),
                           const_shaped_iterator<T>(sA, data_A, sA.size()),
                           data_B + sB.offset(),
                           begin_C, f);
        } else {
            par::transform(const_shaped_iterator<T>(sA, data_A, 0),
                           const_shaped_iterator<T>(sA, data_A, sA.size()),
                           const_shaped_iterator<U>(sB, data_B, 0),
                           begin_C, f);
        }
    }
}

template <typename T, typename U, typename W, typename F>
void transformTo(const Shape& shape_A, const T* data_A, const size_t size_A,
                        const Shape& shape_B, const U* data_B, const size_t size_B,
                        const Shape& shape_C, W* data_C, F f)
{
    if (shape_C.is_contiguous()) {
        transformTo(shape_A, data_A, size_A, shape_B, data_B, size_B, f, shape_C, data_C + shape_C.offset());
    } else {
        transformTo(shape_A, data_A, size_A, shape_B, data_B, size_B, f, shape_C, shaped_iterator<W>(shape_C, data_C, 0));
    }
}
}

template <typename T, typename U, typename W, typename F>
inline Tensor<W>& transformTo(const Tensor<T>& A, const Tensor<U>& B, Tensor<W>& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline TensorView<W>& transformTo(const Tensor<T>& A, const Tensor<U>& B, TensorView<W>& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline TensorView<W>& transformTo(const Tensor<T>& A, const Tensor<U>& B, TensorView<W>&& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline Tensor<W>& transformTo(const Tensor<T>& A, const TensorView<U>& B, Tensor<W>& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline TensorView<W>& transformTo(const Tensor<T>& A, const TensorView<U>& B, TensorView<W>& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline TensorView<W>& transformTo(const Tensor<T>& A, const TensorView<U>& B, TensorView<W>&& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline Tensor<W>& transformTo(const TensorView<T>& A, const Tensor<U>& B, Tensor<W>& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline TensorView<W>& transformTo(const TensorView<T>& A, const Tensor<U>& B, TensorView<W>& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline TensorView<W>& transformTo(const TensorView<T>& A, const Tensor<U>& B, TensorView<W>&& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline Tensor<W>& transformTo(const TensorView<T>& A, const TensorView<U>& B, Tensor<W>& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline TensorView<W>& transformTo(const TensorView<T>& A, const TensorView<U>& B, TensorView<W>& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline TensorView<W>& transformTo(const TensorView<T>& A, const TensorView<U>& B, TensorView<W>&& C, F f) {
    impl::transformTo(A.shape(), A.data(), A.size(), B.shape(), B.data(), B.size(), C.shape(), C.data(), f);
    return C;
}

template <typename T, typename U, typename W, typename F>
inline void transformChannel(const Tensor<T>& A, const Tensor<U>& B, Tensor<W>& C, size_t axis, F fn) {
    impl::transformChannel(A.shape(), A.data(), B.shape(), B.data(), C.shape(), C.begin(), axis, fn);
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

template <typename T, typename U, typename F, typename W = cxx::invoke_result_t<F,T,U>>
inline Tensor<W> transform(const Tensor<T>& A, const TensorView<U>& B, F f) {
    Tensor<W> C(Shape::broadcast(A, B));
    transformTo(A, B, C, f);
    return C;
}

template <typename T, typename U, typename F, typename W = cxx::invoke_result_t<F,T,U>>
inline Tensor<W> transform(const TensorView<T>& A, const Tensor<U>& B, F f) {
    Tensor<W> C(Shape::broadcast(A, B));
    transformTo(A, B, C, f);
    return C;
}

template <typename T, typename U, typename F, typename W = cxx::invoke_result_t<F,T,U>>
inline Tensor<W> transform(const TensorView<T>& A, const TensorView<U>& B, F f) {
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
template <typename U, typename F>
inline Tensor<T>& Tensor<T>::apply(const Tensor<U>& y, F f) {
    return transformTo(*this, y, *this, f);
}

//==-------------------------------------------------------------------------
// Tensor shape operations
//==-------------------------------------------------------------------------

template <typename T>
TensorView<T> broadcast(const Tensor<T>& src, const Shape& shape) {
    return TensorView<T>(src.shape().broadcast(shape), src);
}

template <typename T>
TensorView<T> broadcast(const TensorView<T>& src, const Shape& shape) {
    return TensorView<T>(src.shape().broadcast(shape), src);
}

template <typename T>
void transpose(const Tensor<T>& src, Tensor<T>& dst, const std::vector<size_t>& perm) {
    Shape shape = src.shape().transpose(perm);
    if (shape != dst.shape())
        throw shape_error("transpose: invalid output shape");
    reorder(src, shape, dst);
}

template <typename T>
inline TensorView<T> transpose(const Tensor<T>& src, const std::vector<size_t>& perm) {
    return TensorView<T>(src.shape().transpose(perm), src);
}

template <typename T>
inline TensorView<T> transpose(const Tensor<T>& src) {
    if (src.is_vector()) {
        return TensorView<T>({src.extent(0), 1}, src);
    } else {
        return TensorView<T>(src.shape().transpose(), src);
    }
}

template <typename T>
inline TensorView<T> transpose(const TensorView<T>& src, const std::vector<size_t>& perm) {
    return TensorView<T>(src.shape().transpose(perm), src);
}

template <typename T>
inline TensorView<T> transpose(const TensorView<T>& src) {
    return TensorView<T>(src.shape().transpose(), src);
}

template <typename T>
inline TensorView<T> operator~(const Tensor<T>& src) {
    return transpose(src);
}

template <typename T>
inline TensorView<T> operator~(const TensorView<T>& src) {
    return transpose(src);
}

template <typename T>
inline TensorView<T> slice(const Tensor<T>& src, const std::vector<SliceDim>& dims) {
    return TensorView<T>(src.shape().slice(dims), src);
}

template <typename T>
inline TensorView<T> slice(const TensorView<T>& src, const std::vector<SliceDim>& dims) {
    return TensorView<T>(src.shape().slice(dims), src);
}

template <typename T>
inline TensorView<T> diagonal(const Tensor<T>& src) {
    return TensorView<T>(src.shape().diagonal(), src);
}

template <typename T>
inline TensorView<T> diagonal(const TensorView<T>& src) {
    return TensorView<T>(src.shape().diagonal(), src);
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
        [&](auto r, T sum) {
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
gemv(size_t m, size_t n, const T* A, size_t lda, const T* B, T* C) {
    cblas::gemv(cblas::Layout::RowMajor, cblas::Transpose::NoTrans, m, n, T(1), A, lda, B, 1, T(0), C, 1);
}

template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
gemv(size_t m, size_t n, const T* A, size_t lda, const T* B, T* C) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m, GRAINSIZE), [&](auto r) {
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
gemm(size_t m, size_t n, size_t k, const T* A, size_t lda, const T* B, size_t ldb, T* C, size_t ldc) {
    cblas::gemm(cblas::Layout::RowMajor, cblas::Transpose::NoTrans, cblas::Transpose::NoTrans,
                m, n, k, T(1), A, lda, B, ldb, T(0), C, ldc);
}

template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
gemm(size_t m, size_t n, size_t k, const T* A, size_t lda, const T* B, size_t ldb, T* C, size_t ldc) {
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [&](auto r) {
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
    assert(C->data() != A.data() && C->data() != B.data());

    if (A.is_vector() && B.is_vector()) {
        auto n = A.extent(0);
        assert(n == B.extent(0));
        assert(C->is_vector() && 1 == C->extent(0));
        *C->data() = impl::dot(n, A.data(), B.data());
        return *C;
    }

    if (A.is_matrix() && B.is_vector()) {
        auto m = A.extent(0), n = A.extent(1);
        assert(n == B.extent(0));
        assert(C->is_vector() && m == C->extent(0));
        impl::gemv(m, n, A.data(), A.stride(0), B.data(), C->data());
        return *C;
    }

    if ((A.is_vector() || A.is_matrix()) && B.is_matrix()) {
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
                   A.data(),  A_shape.stride(0),
                   B.data(),  B_shape.stride(0),
                   C->data(), C_shape.stride(0));
        return *C;
    }

    throw std::logic_error("dot: unsupported tensor shape");
}

/**
 * General matrix multiplication.
 */
template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
     const T& beta, Tensor<T>* C,
     bool transA = false, bool transB = false,
     Tensor<T>* = nullptr)
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
     bool transA = false, bool transB = false,
     Tensor<T>* = nullptr)
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
inline size_t gemmWorkspaceSize(
    const Tensor<T>&, const Tensor<T>&, const Tensor<T>&,
    bool = false, bool = false)
{
    // API compatible to DevTensor
    return 0;
}

} // namespace dlf
