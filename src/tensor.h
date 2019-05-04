#ifndef KNERON_TENSOR_H
#define KNERON_TENSOR_H

#include <vector>
#include <iostream>

namespace kneron::model {

/**
 * Tensor is a geometric object that maps in a multi-linear manner geometric
 * vectors, scalars, and other tensors to a resulting tensor.
 *
 * @tparam T the data type of the tensor.
 */
template <typename T>
class Tensor {
    std::vector<size_t> m_dims;
    size_t m_size = 0;
    T* m_data = nullptr;
    std::unique_ptr<T[]> m_alloc_data;

    static size_t sizeOf(const std::vector<size_t>& dims) {
        if (dims.empty())
            return 0;
        size_t size = 1;
        for (auto d : dims)
            size *= d;
        return size;
    }

    static size_t offsetOf(const std::vector<size_t>& dims, const std::initializer_list<size_t>& index) {
        assert(index.size() == dims.size());
        size_t offset = 0, dim = 1;
        auto p = index.end();
        auto q = dims.end();
        while (p != index.begin()) {
            auto i = *--p;
            auto d = *--q;
            offset += dim * i;
            dim *= d;
        }
        return offset;
    }

    static bool nextIndex(const std::vector<size_t>& dims, std::vector<size_t>& index) {
        for (auto i = dims.size(); i != 0; ) {
            --i;
            if (++index[i] < dims[i])
                return true;
            index[i] = 0;
        }
        return false;
    }

    /**
     * Construct a tensor with given dimension and wrapped data. This constructor
     * can only be called from wrap() function.
     *
     * @param dims the tensor dimension
     * @param data the tensor data
     */
    Tensor(std::vector<size_t> dims, T* data)
        : m_dims(std::move(dims))
    {
        m_data = data;
        m_size = sizeOf(m_dims);
    }

    /**
     * Construct a tensor with given dimension and preallocated data. This constructor
     * can only be called from builder functions.
     *
     * @param dims the tensor dimensions.
     * @param data the preallocated tensor data.
     */
    Tensor(std::vector<size_t> dims, std::unique_ptr<T[]> data)
        : m_dims(std::move(dims)), m_alloc_data(std::move(data))
    {
        m_data = m_alloc_data.get();
        m_size = sizeOf(m_dims);
    }

public:
    using element_type = T;

    /**
     * Construct a 0-dimensional tensor.
     */
    Tensor() = default;

    /**
     * Construct a tensor with given dimension.
     *
     * @param dims the tensor dimension
     */
    explicit Tensor(std::vector<size_t> dims)
        : m_dims(std::move(dims))
    {
        m_size = sizeOf(m_dims);
        m_alloc_data = std::make_unique<T[]>(m_size);
        m_data = m_alloc_data.get();
    }

    /**
     * Construct a tensor with given dimension and initial data.
     *
     * @param dims the tensor dimension
     * @param data the tensor data
     */
    Tensor(std::vector<size_t> dims, const std::vector<T>& data)
        : m_dims(std::move(dims))
    {
        m_size = sizeOf(m_dims);
        assert(data.size() == m_size);
        m_alloc_data = std::make_unique<T[]>(m_size);
        m_data = m_alloc_data.get();
        std::copy(data.begin(), data.end(), m_data);
    }

    /**
     * Wraps a raw data as a tensor, given the dimension of tensor.
     *
     * @param dims the tensor dimension
     * @param data the wrapped tensor data.
     */
    static Tensor wrap(std::vector<size_t> dims, T* data) {
        return Tensor(std::move(dims), data);
    }

    /**
     * Build the tensor with given generator function. The generator function
     * accepts the an index and a sequence number which can be aid to generate
     * the tensor data.
     *
     * @param dims the tensor dimension
     * @param f the generator function
     * @return the generated tensor
     */
    static Tensor build(std::vector<size_t> dims, std::function<T(const std::vector<size_t>&, size_t)> f) {
        auto size = sizeOf(dims);
        auto data = std::make_unique<T[]>(size);
        std::vector<size_t> index(dims.size(), 0);
        size_t sequence = 0;

        do {
            data[sequence] = f(index, sequence);
            sequence++;
        } while (nextIndex(dims, index));

        return Tensor(std::move(dims), std::move(data));
    }

    /**
     * Copy constructor.
     */
    Tensor(const Tensor& t)
        : m_dims(t.m_dims), m_size(t.m_size)
    {
        m_alloc_data = std::make_unique<T[]>(m_size);
        m_data = m_alloc_data.get();
        std::copy(t.data(), t.data() + t.size(), m_data);
    }

    /**
     * Copy assignment.
     */
    Tensor& operator=(const Tensor& t) {
        m_dims = t.m_dims;
        if (m_size != t.m_size || m_alloc_data == nullptr) {
            m_size = t.m_size;
            m_alloc_data = std::make_unique<T[]>(m_size);
            m_data = m_alloc_data.get();
        }
        std::copy(t.data(), t.data() + t.size(), m_data);
        return *this;
    }

    /**
     * Move constructor.
     */
    Tensor(Tensor&& t) noexcept
        : m_dims(std::move(t.m_dims)), m_size(t.m_size), m_data(t.m_data), m_alloc_data(std::move(t.m_alloc_data))
    {
        t.m_data = nullptr;
        t.m_size = 0;
    }

    /**
     * Move assignment.
     */
    Tensor& operator=(Tensor&& t) noexcept {
        m_dims = std::move(t.m_dims);
        m_size = t.m_size;
        m_data = t.m_data;
        m_alloc_data = std::move(t.m_alloc_data);
        t.m_data = nullptr;
        t.m_size = 0;
        return *this;
    }

    /**
     * Returns the shape of this tensor.
     */
    const std::vector<size_t>& dims() const noexcept {
        return m_dims;
    }

    /**
     * Returns the total size of this tensor.
     */
    size_t size() const noexcept {
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
    const T operator[](std::initializer_list<size_t> index) const {
        return data()[offsetOf(m_dims, index)];
    }

    /**
     * Returns the mutable element given by the index.
     */
    T& operator[](std::initializer_list<size_t> index) {
        return data()[offsetOf(m_dims, index)];
    }

    /**
     * Equality test for two tensors.
     */
    bool operator==(const Tensor& other) const {
        if (m_dims != other.m_dims)
            return false;
        return std::equal(data(), data()+size(), other.data());
    }

    bool operator!=(const Tensor& other) const {
        return !(*this == other);
    }

    /**
     * Adding two tensors elementwise.
     */
    Tensor operator+(const Tensor& y) const {
        Tensor z(m_dims);
        binop(*this, y, z, std::plus<T>());
        return z;
    }

    /**
     * Inplace add another tensor elementwise.
     */
    Tensor& operator+=(const Tensor& y) {
        binop(*this, y, *this, std::plus<T>());
        return *this;
    }

    /**
     * Adding the tensor with a scalar value.
     */
    Tensor operator+(T v) const {
        Tensor z(m_dims);
        scalarop(data(), v, z.data(), size(), std::plus<T>());
        return z;
    }

    /**
     * Inplace add a scalar value.
     */
    Tensor& operator+=(T v) {
        scalarop(data(), v, data(), size(), std::plus<T>());
        return *this;
    }

    /**
     * Subtracting from another tensor elementwise.
     */
    Tensor operator-(const Tensor& y) const {
        Tensor z(m_dims);
        binop(*this, y, z, std::minus<T>());
        return z;
    }

    /**
     * Inplace subtract another tensor elementwise.
     */
    Tensor& operator-=(const Tensor& y) {
        binop(*this, y, *this, std::minus<T>());
        return *this;
    }

    /**
     * Subtracting the tensor with a scalar value.
     */
    Tensor operator-(T v) const {
        Tensor z(m_dims);
        scalarop(data(), v, z.data(), size(), std::minus<T>());
        return z;
    }

    /**
     * Inplace subtract a scalar value.
     */
    Tensor& operator-=(T v) {
        scalarop(data(), v, data(), size(), std::minus<T>());
        return *this;
    }

    /**
     * Multiply two tensors elementwise.
     */
    Tensor operator*(const Tensor& y) const {
        Tensor z(m_dims);
        binop(*this, y, z, std::multiplies<T>());
        return z;
    }

    /**
     * Inplace multiply another tensor elementwise.
     */
    Tensor& operator*=(const Tensor& y) {
        binop(*this, y, *this, std::multiplies<T>());
        return *this;
    }

    /**
     * Multiply the tensor with a scalar value.
     */
    Tensor operator*(T v) const {
        Tensor z(m_dims);
        scalarop(data(), v, z.data(), size(), std::multiplies<T>());
        return z;
    }

    /**
     * Inplace multiply a scalar value.
     */
    Tensor& operator*=(T v) {
        scalarop(data(), v, data(), size(), std::multiplies<T>());
        return *this;
    }

    /**
     * Divides two tensors elementwise.
     */
    Tensor operator/(const Tensor& y) const {
        Tensor z(m_dims);
        binop(*this, y, z, std::divides<T>());
        return z;
    }

    /**
     * Inplace divides another tensor elementwise.
     */
    Tensor& operator/=(const Tensor& y) {
        binop(*this, y, *this, std::divides<T>());
        return *this;
    }

    /**
     * Divides the tensor with a scalar value.
     */
    Tensor operator/(T v) const {
        Tensor z(m_dims);
        scalarop(data(), v, z.data(), size(), std::divides<T>());
        return z;
    }

    /**
     * Inplace divides a scalar value.
     */
    Tensor& operator/=(T v) {
        scalarop(data(), v, data(), size(), std::divides<T>());
        return *this;
    }

    /**
     * Perform dot product on two matrices.
     */
    Tensor dot(const Tensor& y) const;

    /**
     * Transpose a matrix.
     */
    Tensor transpose() const;

    /**
     * Apply a function on tensor's elements.
     */
    template <typename F>
    Tensor& apply(F f) {
        std::transform(data(), data()+size(), data(), f);
        return *this;
    }

    /**
     * Transform tensor's elements to a new tensor by applying the given function.
     * The element type may change during transformation.
     *
     * @param f the function to be applied.
     * @return the Tensor that contains transformed elements.
     */
    template <typename F, typename U = std::result_of_t<F(T)>>
    Tensor<U> transform(F f) const {
        Tensor<U> res(m_dims);
        std::transform(data(), data()+size(), res.data(), f);
        return res;
    }

    /**
     * Transform tensor's elements to another tensor by applying the given function.
     * The two tensor must have the same shape.
     *
     * @param target the target tensor to store transformed data
     * @param f the function to apply the transformation
     */
    template <typename U, typename F>
    void transformTo(Tensor<U>& target, F f) const {
        assert(dims() == target.dims());
        std::transform(data(), data()+size(), target.data(), f);
    }

    /**
     * Casting element type.
     *
     * @tparam U the target element type
     * @return the Tensor with new element type.
     */
    template <typename U>
    Tensor<U> cast() const {
        return transform([](T x) { return static_cast<U>(x); });
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        printRec(os, t.dims(), 0, t.data());
        return os;
    }

private:
    template <typename F>
    static void binop(const Tensor& x, const Tensor& y, Tensor& z, F f) {
        assert(x.dims() == y.dims());
        std::transform(x.data(), x.data()+x.size(), y.data(), z.data(), f);
    }

    template <typename F>
    static void scalarop(const T* x, T y, T* z, size_t n, F f) {
        for (size_t i = 0; i < n; i++) {
            z[i] = f(x[i], y);
        }
    }

    static const T* printRec(std::ostream& out, const std::vector<size_t>& dims, size_t level, const T* data);
};

/**
 * Perform binary operation on two tensors elementwise.
 */
template <typename T, typename F>
inline Tensor<T> transform(const Tensor<T>& x, const Tensor<T>& y, F f) {
    assert(x.dims() == y.dims());
    Tensor<T> z(x.dims());
    std::transform(x.data(), x.data()+x.size(), y.data(), z.data(), f);
    return z;
}

/**
 * Perform binary operation on two tensors element and store the result into
 * the third tensor.
 */
template <typename T, typename F>
inline void transformTo(Tensor<T>& z, const Tensor<T>& x, const Tensor<T>& y, F f) {
    assert(x.dims() == y.dims() && y.dims() == z.dims());
    std::transform(x.data(), x.data()+x.size(), y.data(), z.data(), f);
}

template <typename T>
Tensor<T> Tensor<T>::dot(const Tensor& y) const {
    assert(dims().size() == 2 && y.dims().size() == 2);

    auto n = m_dims[0];
    auto p = m_dims[1];
    auto m = y.m_dims[1];
    assert(p == y.m_dims[0]);

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
    assert(m_dims.size() == 2);

    auto n = m_dims[0];
    auto m = m_dims[1];

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
const T* Tensor<T>::printRec(std::ostream& out, const std::vector<size_t>& dims, size_t level, const T* data) {
    auto d = dims[level];

    out << '[';
    if (level == dims.size()-1) {
        // last level, printing data
        for (int i = 0; i < d; i++) {
            out << *data++;
            if (i < d-1)
                out << ", ";
        }
    } else {
        // intermediate levels, recursive
        for (int i = 0; i < d; i++) {
            data = printRec(out, dims, level+1, data);
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
