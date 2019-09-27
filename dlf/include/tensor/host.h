#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor declaration
//==-------------------------------------------------------------------------

template <typename T> class TensorView;

template <typename T, typename D>
using is_random_distribution_type = cxx::conjunction<
    std::is_same<T, typename D::result_type>,
    std::is_same<D, typename D::param_type::distribution_type>>;

/**
 * Tensor is a geometric object that maps in a multi-linear manner geometric
 * vectors, scalars, and other tensors to a resulting tensor.
 *
 * @tparam T the data type of the tensor.
 */
template <typename T>
class Tensor : public Spatial<Tensor<T>> {
    T* m_data = nullptr;
    std::shared_ptr<T> m_alloc_data;

    void init() {
        assert(size() != 0);
        if (std::is_trivially_destructible<T>::value)
            m_alloc_data = std::shared_ptr<T>(new T[size()]);
        else
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
     * @param dist the random distribution.
     */
    template <typename D>
    std::enable_if_t<is_random_distribution_type<T, D>::value, Tensor&>
    random(D&& d) &;

    template <typename D>
    std::enable_if_t<is_random_distribution_type<T, D>::value, Tensor>
    random(D&& d) &&;

    /**
     * Fill the tensor with random data with uniform distribution.
     *
     * @param low the lowest random value
     * @param high the highest random value
     */
    Tensor& random(T low = 0, T high = std::numeric_limits<T>::max()) &;
    Tensor random(T low = 0, T high = std::numeric_limits<T>::max()) &&;

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
    using Spatial<Tensor>::shape;
    using Spatial<Tensor>::size;

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

    T& operator*() noexcept { return *m_data; }
    const T& operator*() const noexcept { return *m_data; }

    /**
     * Returns a view of this tensor.
     */
    TensorView<T> view() const {
        return TensorView<T>(shape(), *this);
    }

    /**
     * Returns a view of this tensor with given shape.
     */
    TensorView<T> view(Shape shape) const {
        return TensorView<T>(std::move(shape), *this);
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
    using Spatial<Tensor>::reshape;
    using Spatial<Tensor>::flatten;
    using Spatial<Tensor>::squeeze;
    using Spatial<Tensor>::unsqueeze;
};

template <typename T>
class TensorView : public Spatial<TensorView<T>> {
    Shape m_original_shape;
    T* m_data;
    std::shared_ptr<T> m_alloc_data;

public:
    TensorView() = default;
    TensorView(Shape shape, const Tensor<T>& src);
    TensorView(Shape shape, const TensorView<T>& src);

    TensorView& resize(const Shape& shape) {
        if (this->shape() != shape)
            throw shape_error("incompatible shape");
        return *this;
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, TensorView&>
    resize(Args... args) {
        return resize({static_cast<size_t>(args)...});
    }

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
    using Spatial<TensorView>::shape;
    using Spatial<TensorView>::size;

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

    T& operator*() noexcept { return m_data[shape().offset()]; }
    const T& operator*() const noexcept { return m_data[shape().offset()]; }

    TensorView view() const {
        return *this;
    }

    TensorView view(Shape shape) const {
        return TensorView(std::move(shape), *this);
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

    template <typename D>
    std::enable_if_t<is_random_distribution_type<T, D>::value, TensorView&>
    random(D&& d);

    TensorView& random(T low = 0, T high = std::numeric_limits<T>::max());
};

//==-------------------------------------------------------------------------
// Tensor constructors
//==-------------------------------------------------------------------------

template <typename T>
Tensor<T>::Tensor(Shape shape) : Spatial<Tensor>(std::move(shape)) {
    init();
}

template <typename T>
Tensor<T>::Tensor(Shape shape, const T& initial) : Spatial<Tensor>(std::move(shape)) {
    init();
    fill(initial);
}

template <typename T>
Tensor<T>::Tensor(Shape shape, T* data) : Spatial<Tensor>(std::move(shape)) {
    m_data = data;
}

template <typename T>
template <typename It>
Tensor<T>::Tensor(Shape shape, It begin, RequireInputIterator<It> end)
    : Spatial<Tensor>(std::move(shape))
{
    init();
    assert(std::distance(begin, end) == size());
    std::copy(begin, end, m_data);
}

template <typename T>
Tensor<T>::Tensor(Shape shape, std::initializer_list<T> il)
    : Spatial<Tensor>(std::move(shape))
{
    init();
    assert(size() == il.size());
    std::copy(il.begin(), il.end(), m_data);
}

template <typename T>
Tensor<T>::Tensor(Shape shape, std::shared_ptr<T> data)
    : Spatial<Tensor>(std::move(shape)), m_alloc_data(std::move(data))
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
    Tensor<T> ret{Shape()};
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
Tensor<T>::Tensor(const Tensor& t) : Spatial<Tensor>(t) {
    init();
    std::copy(t.begin(), t.end(), m_data);
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& t) {
    auto old_size = size();
    Spatial<Tensor>::set_shape(t.shape());
    if (size() != old_size || m_alloc_data == nullptr)
        init();
    std::copy(t.begin(), t.end(), m_data);
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const TensorView<T>& v) {
    auto old_size = size();
    Spatial<Tensor>::set_shape(v.shape());
    if (size() != old_size || m_alloc_data == nullptr)
        init();
    reorder(v, *this);
    return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor&& t) noexcept
    : Spatial<Tensor>(std::move(t)),
      m_data(std::exchange(t.m_data, nullptr)),
      m_alloc_data(std::move(t.m_alloc_data))
{
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& t) noexcept {
    Spatial<Tensor>::set_shape(t.shape());
    m_data = std::exchange(t.m_data, nullptr);
    m_alloc_data = std::move(t.m_alloc_data);
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::resize(const Shape& shape) {
    if (this->empty()) {
        Spatial<Tensor>::set_shape(shape);
        init();
    } else if (this->shape() != shape) {
        throw shape_error("incompatible shape");
    }
    return *this;
}

//==-------------------------------------------------------------------------
// Nested initializer list
//==-------------------------------------------------------------------------

template <typename T, size_t N>
struct nested_initializer_list {
    using type = std::initializer_list<typename nested_initializer_list<T, N-1>::type>;
};

template <typename T>
struct nested_initializer_list<T, 0> {
    using type = T;
};

template <typename T, size_t N>
using nested_initializer_list_t = typename nested_initializer_list<T, N>::type;

template <typename T, size_t N>
struct NestedInitializerListProcessor {
    static void get_shape(std::vector<size_t>& shape, nested_initializer_list_t<T, N> list) {
        int i = shape.size() - N;
        shape[i] = std::max(shape[i], list.size());
        for (auto nested : list) {
            NestedInitializerListProcessor<T, N-1>::get_shape(shape, nested);
        }
    }

    static void copy_data(T* data, const std::vector<size_t>& shape, nested_initializer_list_t<T, N> list) {
        int i = shape.size() - N;
        size_t stride = 1;
        for (int j = i+1; j < shape.size(); j++)
            stride *= shape[j];
        for (auto nested : list) {
            NestedInitializerListProcessor<T, N-1>::copy_data(data, shape, nested);
            if (nested.size() < shape[i+1])
                std::fill(data + stride/shape[i+1]*nested.size(),
                          data + stride,
                          T{});
            data += stride;
        }
    }
};

template <typename T>
struct NestedInitializerListProcessor<T, 1> {
    static void get_shape(std::vector<size_t>& shape, std::initializer_list<T> list) {
        int i = shape.size() - 1;
        shape[i] = std::max(shape[i], list.size());
    }

    static void copy_data(T* data, const std::vector<size_t>& shape, std::initializer_list<T> list) {
        int i = shape.size() - 1;
        std::copy(list.begin(), list.end(), data);
        if (list.size() < shape[i])
            std::fill(data + list.size(), data + shape[i], T{});
    }
};

template <typename T>
struct NestedInitializerListProcessor<T, 0> {
    static void get_shape(std::vector<size_t>&, T) {}
    static void copy_data(T* data, const std::vector<size_t>&, T value) {
        *data = value;
    }
};

/**
 * Construct a tensor with a nested initializer list. The tensor shape is inferred
 * from nested initializer list.
 *
 * @param init the nested initializer list
 */
template <typename T, size_t N>
Tensor<T> make_tensor(nested_initializer_list_t<T, N> list) {
    std::vector<size_t> dims(N);
    NestedInitializerListProcessor<T, N>::get_shape(dims, list);

    Tensor<T> res{Shape{dims}};
    NestedInitializerListProcessor<T, N>::copy_data(res.data(), dims, list);
    return res;
}

template <typename T>
inline Tensor<T> Scalar(T value) {
    return Tensor<T>::scalar(value);
}

template <typename T>
inline Tensor<T> Vector(std::initializer_list<T> list) {
    return Tensor<T>(Shape(list.size()), list);
}

template <typename T>
inline Tensor<T> Matrix(std::initializer_list<std::initializer_list<T>> list) {
    return make_tensor<T, 2>(list);
}

//==-------------------------------------------------------------------------
// TensorView implementation
//==-------------------------------------------------------------------------

template <typename T>
TensorView<T>::TensorView(Shape shape, const Tensor<T>& src)
    : Spatial<TensorView>(std::move(shape), true),
      m_original_shape(src.original_shape()),
      m_data(src.m_data),
      m_alloc_data(src.m_alloc_data)
{}

template <typename T>
TensorView<T>::TensorView(Shape shape, const TensorView<T>& src)
    : Spatial<TensorView>(std::move(shape), true),
      m_original_shape(src.original_shape()),
      m_data(src.m_data),
      m_alloc_data(src.m_alloc_data)
{}

//==-------------------------------------------------------------------------
// Tensor randomize
//==-------------------------------------------------------------------------

namespace detail {
template <typename T, typename TensorT, typename D>
inline TensorT& randomize(TensorT& t, D&& d) {
    std::random_device rd;
    std::mt19937 eng(rd());
    return t.generate(std::bind(std::forward<D>(d), eng));
}

template <typename TensorT, typename T>
std::enable_if_t<std::is_integral<T>::value, TensorT&>
inline randomize(TensorT& t, T low, T high) {
    std::random_device rd;
    std::mt19937 eng(rd());
    return t.generate(std::bind(std::uniform_int_distribution<T>(low, high), eng));
}

template <typename TensorT, typename T>
std::enable_if_t<std::is_floating_point<T>::value, TensorT&>
inline randomize(TensorT& t, T low, T high) {
    std::random_device rd;
    std::mt19937 eng(rd());
    return t.generate(std::bind(std::uniform_real_distribution<T>(low, high), eng));
}
} // namespace detail

template <typename T>
template <typename D>
std::enable_if_t<is_random_distribution_type<T, D>::value, Tensor<T>&>
inline Tensor<T>::random(D&& d) & {
    return detail::randomize<T>(*this, std::forward<D>(d));
}

template <typename T>
template <typename D>
std::enable_if_t<is_random_distribution_type<T, D>::value, Tensor<T>>
inline Tensor<T>::random(D&& d) && {
    return detail::randomize<T>(*this, std::forward<D>(d));
}

template <typename T>
inline Tensor<T>& Tensor<T>::random(T low, T high) & {
    return detail::randomize(*this, low, high);
}

template <typename T>
inline Tensor<T> Tensor<T>::random(T low, T high) && {
    return std::move(detail::randomize(*this, low, high));
}

template <typename T>
template <typename D>
std::enable_if_t<is_random_distribution_type<T, D>::value, TensorView<T>&>
inline TensorView<T>::random(D&& d) {
    return detail::randomize(*this, std::forward<D>(d));
}

template <typename T>
inline TensorView<T>& TensorView<T>::random(T low, T high) {
    return detail::randomize(*this, low, high);
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
        if (!t.empty()) {
            if (t.is_scalar()) {
                out << *t;
            } else {
                out << t.shape() << '\n';
                print_rec(out, w, t.shape(), 0, t.begin());
                out << '\n';
            }
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
    transformTo(*this, *this, f);
    return *this;
}

template <typename T>
template <typename F>
inline TensorView<T>& TensorView<T>::apply(F f) {
    transformTo(*this, *this, f);
    return *this;
}

template <typename T>
template <typename U, typename F>
inline Tensor<T>& Tensor<T>::apply(const Tensor<U>& y, F f) {
    transformTo(*this, y, *this, f);
    return *this;
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
