#pragma once

#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

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
        m_alloc_data = std::shared_ptr<T>(new T[size()], std::default_delete<T[]>());
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
     * Construct a tensor with given dimensions and fill with constant value.
     *
     * @param shape then tensor shape
     * @param initial the initial value
     */
    explicit Tensor(Shape shape, const T& initial);

    /**
     * Construct a tensor with input iterator denoted by [begin,end).
     *
     * @param shape the tensor dimensions
     * @param begin the start of input iterator
     * @param end the end of input iterator
     */
    template <typename It>
    explicit Tensor(Shape shape, It begin, RequireInputIterator<It> end);

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
    explicit Tensor(Shape shape, std::shared_ptr<T> data);

    /**
     * Construct a tensor from a tensor view. The contents of the tensor view is
     * copied into newly created tensor and the shape is normalized.
     *
     * @param view the tensor view
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
     * Create an identity tensor.
     *
     * @param r the tensor rank
     * @param n the tensor dimension
     * @param value the identity value
     * @return the identity tensor
     */
    static Tensor identity(Shape shape, const T& value = T{1});

    /**
     * Create a tensor with values starting from n.
     *
     * @param shape the tensor dimension
     * @param n the starting value in the tensor data.
     * @param step the increment step
     */
    static Tensor range(Shape shape, T n = T{0}, T step = T{1});

    /**
     * Fill tensor with generator function.
     *
     * @param f the generator function
     */
    template <typename F>
    Tensor& generate(F f) &;

    template <typename F>
    Tensor generate(F f) &&;

    /**
     * Fill the tensor with a scalar value.
     *
     * @param value the constant scalar value.
     */
    Tensor& fill(const T& value) &;
    Tensor fill(const T& value) &&;

    /**
     * Fill the tensor with random data.
     *
     * @param low the lowest random value
     * @param high the highest random value
     */
    Tensor& random(T low, T high) &;
    Tensor random(T low, T high) &&;

    // Copy and move constructors/assignments.
    Tensor(const Tensor& t);
    Tensor& operator=(const Tensor& t);
    Tensor(Tensor&& t) noexcept;
    Tensor& operator=(Tensor&& t) noexcept;

    Tensor& operator=(const TensorView<T>& v);

    /**
     * Allocate tensor data if necessary.
     *
     * This tensor must be an uninitialized tensor or initialized with the given
     * shape. In all other cases, the shape_error exception is thrown.
     */
    Tensor& resize(const Shape& shape);

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, Tensor&>
    resize(Args... args) {
        return resize({static_cast<size_t>(args)...});
    }

public: // Attributes
    /**
     * The original shape is same as this tensor's shape.
     */
    const Shape& original_shape() const {
        return shape();
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
    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, const T&>
    operator()(Args... args) const noexcept {
        return data()[shape().offset({static_cast<size_t>(args)...})];
    }

    /**
     * Returns the mutable element given by the index.
     */
    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, T&>
    operator()(Args... args) noexcept {
        return data()[shape().offset({static_cast<size_t>(args)...})];
    }

    /**
     * Returns a slice given by the index.
     */
    Tensor<T> operator[](int index);

    /**
     * Returns a view of this tensor.
     */
    TensorView<T> view() const {
        return TensorView<T>(shape(), *this);
    }

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

public: // Shape operations
    using Shaped::reshape;
    using Shaped::flatten;
    using Shaped::squeeze;
    using Shaped::unsqueeze;

    TensorView<T> broadcast(const Shape& to) const {
        return TensorView<T>(shape().broadcast(to), *this);
    }

    TensorView<T> transpose(const std::vector<size_t>& perm) const {
        return TensorView<T>(shape().transpose(perm), *this);
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, TensorView<T>>
    transpose(Args... args) const {
        return transpose({static_cast<size_t>(args)...});
    }

    TensorView<T> transpose() const {
        return TensorView<T>(shape().transpose(), *this);
    }

    TensorView<T> operator~() const {
        return transpose();
    }

    TensorView<T> slice(
        const std::vector<int>& starts, const std::vector<int>& ends,
        const std::vector<int>& axes, const std::vector<int>& steps) const
    {
        return TensorView<T>(shape().slice(starts, ends, axes, steps), *this);
    }

    TensorView<T> slice(const std::vector<SliceDim>& dims) const {
        return TensorView<T>(shape().slice(dims), *this);
    }

    TensorView<T> operator[](const std::vector<SliceDim>& dims) const {
        return slice(dims);
    }

    TensorView<T> slice(const char* spec) const {
        return TensorView<T>(shape().slice(spec), *this);
    }

    TensorView<T> operator[](const char* spec) const {
        return slice(spec);
    }

    TensorView<T> diagonal(int offset = 0, int axis1 = -2, int axis2 = -1) const {
        return TensorView<T>(shape().diagonal(offset, axis1, axis2), *this);
    }
};

template <typename T>
class TensorView : public Shaped {
    Shape m_original_shape;
    T* m_data;
    std::shared_ptr<T> m_alloc_data;

public:
    TensorView() = default;
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
    const Shape& original_shape() const {
        return m_original_shape;
    }

    T* data() noexcept { return m_data; }
    const T* data() const noexcept { return m_data; }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, const T&>
    operator()(Args... args) const noexcept {
        return data()[shape().offset({static_cast<size_t>(args)...})];
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, T&>
    operator()(Args... args) noexcept {
        return data()[shape().offset({static_cast<size_t>(args)...})];
    }

    /**
     * Returns a deep copy of this view.
     */
    Tensor<T> copy() const {
        return Tensor<T>(*this);
    }

    /**
     * Returns a shallow copy of the view if the view is contiguous, otherwise,
     * a deep copy is returned.
     */
    Tensor<T> reorder() const {
        if (shape().is_contiguous()) {
            Tensor<T> res(shape(), m_alloc_data);
            res.m_data = m_data + shape().offset();
            return res;
        } else {
            return copy();
        }
    }

    operator Tensor<T>() const {
        return reorder();
    }

public: // Operations
    template <typename F>
    TensorView& apply(F f);

    template <typename U>
    Tensor<U> cast() const;

    TensorView& fill(const T& value) {
        std::fill(begin(), end(), value);
        return *this;
    }

    template <typename F>
    TensorView& generate(F f) {
        std::generate(begin(), end(), f);
        return *this;
    }

    TensorView& random(T low, T high);

public: // Shape operations
    TensorView<T> broadcast(const Shape& to) const {
        return TensorView<T>(shape().broadcast(to), *this);
    }

    TensorView<T> transpose(const std::vector<size_t>& perm) const {
        return TensorView<T>(shape().transpose(perm), *this);
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, TensorView<T>>
    transpose(Args... args) const {
        return transpose({static_cast<size_t>(args)...});
    }

    TensorView<T> transpose() const {
        return TensorView<T>(shape().transpose(), *this);
    }

    TensorView<T> operator~() const {
        return transpose();
    }

    TensorView<T> slice(
        const std::vector<int>& starts, const std::vector<int>& ends,
        const std::vector<int>& axes, const std::vector<int>& steps) const
    {
        return TensorView<T>(shape().slice(starts, ends, axes, steps), *this);
    }

    TensorView<T> slice(const std::vector<SliceDim>& dims) const {
        return TensorView<T>(shape().slice(dims), *this);
    }

    TensorView<T> operator[](const std::vector<SliceDim>& dims) const {
        return slice(dims);
    }

    TensorView<T> slice(const char* spec) const {
        return TensorView<T>(shape().slice(spec), *this);
    }

    TensorView<T> operator[](const char* spec) const {
        return slice(spec);
    }

    TensorView<T> diagonal(int offset = 0, int axis1 = -2, int axis2 = -1) const {
        return TensorView<T>(shape().diagonal(offset, axis1, axis2), *this);
    }
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
Tensor<T>::Tensor(Shape shape, const T& initial)
    : Shaped(std::move(shape))
{
    init();
    fill(initial);
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
Tensor<T>::Tensor(const TensorView<T>& view) {
    reorder(view, *this);
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
Tensor<T> Tensor<T>::identity(Shape shape, const T& value) {
    Tensor res(std::move(shape), T{});
    res.diagonal().fill(value);
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
template <typename F>
inline Tensor<T>& Tensor<T>::generate(F f) & {
    std::generate(begin(), end(), f);
    return *this;
}

template <typename T>
template <typename F>
inline Tensor<T> Tensor<T>::generate(F f) && {
    std::generate(begin(), end(), f);
    return std::move(*this);
}

template <typename T>
inline Tensor<T>& Tensor<T>::fill(const T& value) & {
    std::fill(begin(), end(), value);
    return *this;
}

template <typename T>
inline Tensor<T> Tensor<T>::fill(const T& value) && {
    std::fill(begin(), end(), value);
    return std::move(*this);
}

template <typename T>
Tensor<T>::Tensor(const Tensor& t) : Shaped(t) {
    init();
    std::copy(t.begin(), t.end(), m_data);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& t) {
    auto old_size = size();
    Shaped::resize(t.shape());
    if (size() != old_size || m_alloc_data == nullptr)
        init();
    std::copy(t.begin(), t.end(), m_data);
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const TensorView<T>& v) {
    auto old_size = size();
    Shaped::resize(v.shape());
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
    Shaped::resize(t.shape());
    m_data = std::exchange(t.m_data, nullptr);
    m_alloc_data = std::move(t.m_alloc_data);
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::resize(const Shape& shape) {
    if (empty()) {
        Shaped::resize(shape);
        init();
    } else if (this->shape() != shape) {
        throw shape_error("incompatible shape");
    }
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

template <typename T>
inline Tensor<T> Tensor<T>::operator[](int index) {
    auto slice_shape = shape().slice({{index, index+1}});
    return wrap(slice_shape, data() + slice_shape.offset());
}

//==-------------------------------------------------------------------------
// TensorView implementation
//==-------------------------------------------------------------------------

template <typename T>
TensorView<T>::TensorView(Shape shape, const Tensor<T>& src)
    : Shaped(std::move(shape), true),
      m_original_shape(src.original_shape()),
      m_data(src.m_data),
      m_alloc_data(src.m_alloc_data)
{}

template <typename T>
TensorView<T>::TensorView(Shape shape, const TensorView<T>& src)
    : Shaped(std::move(shape), true),
      m_original_shape(src.original_shape()),
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

//==-------------------------------------------------------------------------
// Tensor randomize
//==-------------------------------------------------------------------------

template <typename T>
class randomize_detail {
    template <typename TensorT>
    static TensorT& randomize(TensorT& t, T low, T high);

    friend class Tensor<T>;
    friend class TensorView<T>;
};

template <typename T>
template <typename TensorT>
TensorT& randomize_detail<T>::randomize(TensorT& t, T low, T high) {
    static_assert(std::is_integral<T>::value, "randomize: requires integral type");
    std::random_device rd;
    std::default_random_engine eng(rd());
    return t.generate(std::bind(std::uniform_int_distribution<T>(low, high), eng));
}

template <>
template <typename TensorT>
TensorT& randomize_detail<float>::randomize(TensorT& t, float low, float high) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    return t.generate(std::bind(std::uniform_real_distribution<float>(low, high), eng));
}

template <>
template <typename TensorT>
TensorT& randomize_detail<double>::randomize(TensorT& t, double low, double high) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    return t.generate(std::bind(std::uniform_real_distribution<double>(low, high), eng));
}

template <typename T>
inline Tensor<T>& Tensor<T>::random(T low, T high) & {
    return randomize_detail<T>::randomize(*this, low, high);
}

template <typename T>
inline Tensor<T> Tensor<T>::random(T low, T high) && {
    return std::move(randomize_detail<T>::randomize(*this, low, high));
}

template <typename T>
inline TensorView<T>& TensorView<T>::random(T low, T high) {
    return randomize_detail<T>::randomize(*this, low, high);
}

//==-------------------------------------------------------------------------
// Tensor printer
//==-------------------------------------------------------------------------

class tensor_printer {
    template <typename Iterator>
    static Iterator print_rec(std::ostream& out, int w, const Shape& shape, size_t level, Iterator cur) {
        auto d = shape.extent(level);

        if (level == shape.rank()-1) {
            // last level, printing data
            out << '[';
            for (int i = 0; ; i++) {
                out << std::setw(w) << *cur++;
                if (i == d-1)
                    break;
                out << ',';
            }
            out << ']';
        } else {
            // Intermediate levels, recursive
            out << '[';
            for (int i = 0; ; i++) {
                cur = print_rec(out, w, shape, level+1, cur);
                if (i == d-1)
                    break;
                out << ',' << '\n';
                if (level != shape.rank()-2)
                    out << '\n';
                for (int j = 0; j <= level; j++)
                    out << ' ';
            }
            out << ']';
        }

        return cur;
    }

    template <typename TensorT>
    static std::ostream& print(std::ostream& out, const TensorT& t) {
        auto w = out.width(0);
        out << t.shape() << '\n';
        if (!t.empty()) {
            print_rec(out, w, t.shape(), 0, t.begin());
            out << '\n';
        }
        return out;
    }

    template <typename T>
    friend std::ostream& operator<<(std::ostream& out, const Tensor<T>& t);

    template <typename T>
    friend std::ostream& operator<<(std::ostream& out, const TensorView<T>& t);
};

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const Tensor<T>& t) {
    return tensor_printer::print(out, t);
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const TensorView<T>& v) {
    return tensor_printer::print(out, v);
}

//==-------------------------------------------------------------------------
// Tensor operations
//==-------------------------------------------------------------------------

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
template <typename U>
inline Tensor<U> TensorView<T>::cast() const {
    return transform(*this, [](const T& x) { return static_cast<U>(x); });
}

template <typename T>
inline void flat_copy(const Tensor<T>& src, Tensor<T>& dst) {
    assert(src.size() == dst.size());
    if (src.data() != dst.data()) {
        par::copy(src.begin(), src.end(), dst.begin());
    }
}

} // namespace dlf
