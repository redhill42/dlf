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
    size_t extent(size_t dim) const noexcept {
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

    // Tensor operators
#define DECLARE_OPERATOR(op) \
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U,T>>> \
    Tensor& operator op(const Tensor<U>& y); \
    Tensor& operator op(const T& b);

    DECLARE_OPERATOR(+=)
    DECLARE_OPERATOR(-=)
    DECLARE_OPERATOR(*=)
    DECLARE_OPERATOR(/=)
#undef DECLARE_OPERATOR

    /**
     * Perform dot product on two matrices. A matrix is a 2-dimension tensor.
     */
    Tensor dot(const Tensor& y) const;

    /**
     * Transpose a matrix. A matrix is a 2-dimension tensor.
     */
    Tensor transpose() const;

    /**
     * Transpose a matrix into target.
     */
    void transposeTo(Tensor& target) const;

    /**
     * Apply a unary function on tensor's elements.
     */
    template <typename F>
    Tensor& apply(F f);

    /**
     * Apply the function pointer on tensor's elements. This is a workaround for
     * overloaded function deduction issue.
     */
    Tensor& apply(T(*f)(T));

    /**
     * Apply a binary function on two tensor's elements.
     */
    template <typename U, typename F>
    Tensor& apply(const Tensor<U>& y, F f);

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
     * Transform two tensor's elements to a new tensor by applying the given
     * binary function. The element type may change during transformation.
     *
     * @param y another tensor involved in transformation
     * @param f the function to be applied.
     * @return the Tensor that contains transformed elements.
     */
    template <typename U, typename F, typename W = std::result_of_t<F(T,U)>>
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
Tensor<T>::Tensor(Shape shape, It begin, RequireInputIterator<It> end)
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
inline Tensor<T>::Tensor(Shape shape, std::initializer_list<T> init)
    : Tensor(std::move(shape), init.begin(), init.end())
{
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
    assert(index < m_shape.extent(0));

    auto slice_shape = m_shape.shrink();
    auto slice_size = slice_shape.size();
    auto slice_data = data() + index * slice_size;
    return wrap(slice_shape, slice_data);
}

template <typename T>
template <typename F>
inline Tensor<T>& Tensor<T>::apply(F f) {
    std::transform(begin(), end(), begin(), f);
    return *this;
}

template <typename T>
inline Tensor<T>& Tensor<T>::apply(T(*f)(T)) {
    std::transform(begin(), end(), begin(), f);
    return *this;
}

template <typename T>
template <typename U, typename F>
inline Tensor<T>& Tensor<T>::apply(const Tensor<U>& y, F f) {
    assert(shape() == y.shape());
    std::transform(begin(), end(), y.begin(), begin(), f);
    return *this;
}

template <typename T>
template <typename F, typename U>
Tensor<U> Tensor<T>::transform(F f) const {
    Tensor<U> res(shape());
    std::transform(begin(), end(), res.begin(), f);
    return res;
}

template <typename T>
template <typename U, typename F>
void Tensor<T>::transformTo(Tensor<U>& target, F f) const {
    assert(shape() == target.shape());
    std::transform(begin(), end(), target.begin(), f);
}

template <typename T>
template <typename U, typename F, typename W>
Tensor<W> Tensor<T>::transform(const Tensor<U>& y, F f) const {
    assert(shape() == y.shape());
    Tensor<W> z(shape());
    std::transform(begin(), end(), y.begin(), z.begin(), f);
    return z;
}

template <typename T>
template <typename U, typename W, typename F>
void Tensor<T>::transformTo(Tensor<W>& z, const Tensor<U>& y, F f) const {
    assert(shape() == y.shape() && shape() == z.shape());
    std::transform(begin(), end(), y.begin(), z.begin(), f);
}

template <typename T>
template <typename U>
inline Tensor<U> Tensor<T>::cast() const {
    return transform([](const T& x) { return static_cast<U>(x); });
}

#define DEFINE_OPERATOR(op) \
    template <typename T> \
    template <typename U, typename> \
    inline Tensor<T>& Tensor<T>::operator op##=(const Tensor<U>& y) { \
        return apply(y, [](const T& a, const U& b){return a op b;}); \
    } \
    template <typename T> \
    inline Tensor<T>& Tensor<T>::operator op##=(const T& b) { \
        return apply([&b](const T& a){return a op b;}); \
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
    }

DEFINE_OPERATOR(+)
DEFINE_OPERATOR(-)
DEFINE_OPERATOR(*)
DEFINE_OPERATOR(/)

#undef DEFINE_OPERATOR

template <typename T>
Tensor<T> operator-(const Tensor<T>& x) {
    return x.transform([](const T& a){return -a;});
}

template <typename T>
Tensor<T> Tensor<T>::dot(const Tensor& y) const {
    assert(is_matrix() && y.is_matrix());

    auto n = shape().extent(0);
    auto p = shape().extent(1);
    auto m = y.shape().extent(1);
    assert(p == y.shape().extent(0));

    Tensor z({n, m});
    auto px = data();
    auto py = y.data();
    auto pz = z.data();
    int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            T v{};
            for (k = 0; k < p; k++)
                v += px[i * p + k] * py[k * m + j];
            pz[i * m + j] = v;
        }
    }

    return z;
}

template <typename T>
void Tensor<T>::transposeTo(Tensor& target) const {
    assert(is_matrix() && target.is_matrix());
    assert(shape().extent(0) == target.shape().extent(1));
    assert(shape().extent(1) == target.shape().extent(0));

    auto n = shape().extent(0);
    auto m = shape().extent(1);
    auto px = data();
    auto py = target.data();
    int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            *py++ = px[j * m + i];
    }
}

template <typename T>
inline Tensor<T> Tensor<T>::transpose() const {
    assert(is_matrix());
    Tensor res({shape().extent(1), shape().extent(0)});
    transposeTo(res);
    return res;
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
    out << ']';

    return data;
}

} // namespace kneron::model

#endif //KNERON_TENSOR_H
