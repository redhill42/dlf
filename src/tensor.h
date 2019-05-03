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
            size_t i = *--p;
            size_t d = *--q;
            offset += dim * i;
            dim *= d;
        }
        return offset;
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
    static Tensor<T> wrap(std::vector<size_t> dims, T* data) {
        return Tensor<T>(std::move(dims), data);
    }

    /**
     * Copy constructor.
     */
    Tensor(const Tensor<T>& t)
        : m_dims(t.m_dims), m_size(t.m_size)
    {
        m_alloc_data = std::make_unique<T[]>(m_size);
        m_data = m_alloc_data.get();
        std::copy(t.data(), t.data() + t.size(), m_data);
    }

    /**
     * Copy assignment.
     */
    Tensor<T>& operator=(const Tensor<T>& t) {
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
    Tensor(Tensor<T>&& t) noexcept
        : m_dims(std::move(t.m_dims)), m_size(t.m_size), m_data(t.m_data), m_alloc_data(std::move(t.m_alloc_data))
    {
        t.m_data = nullptr;
        t.m_size = 0;
    }

    /**
     * Move assignment.
     */
    Tensor<T>& operator=(Tensor<T>&& t) noexcept {
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
    bool operator==(const Tensor<T>& other) const {
        if (m_dims != other.m_dims)
            return false;
        return std::equal(data(), data()+size(), other.data());
    }

    bool operator!=(const Tensor<T>& other) const {
        return !(*this == other);
    }

    /**
     * Adding two tensors elementwise.
     */
    Tensor<T> operator+(const Tensor<T>& y) const {
        Tensor<T> z(m_dims);
        binop(*this, y, z, std::plus<T>());
        return z;
    }

    /**
     * Inplace add another tensor elementwise.
     */
    Tensor<T>& operator+=(const Tensor<T>& y) {
        binop(*this, y, *this, std::plus<T>());
        return *this;
    }

    /**
     * Adding the tensor with a scalar value.
     */
    Tensor<T> operator+(T v) const {
        Tensor<T> z(m_dims);
        scalarop(data(), v, z.data(), size(), std::plus<T>());
        return z;
    }

    /**
     * Inplace add a scalar value.
     */
    Tensor<T>& operator+=(T v) {
        scalarop(data(), v, data(), size(), std::plus<T>());
        return *this;
    }

    /**
     * Subtracting from another tensor elementwise.
     */
    Tensor<T> operator-(const Tensor<T>& y) const {
        Tensor<T> z(m_dims);
        binop(*this, y, z, std::minus<T>());
        return z;
    }

    /**
     * Inplace subtract another tensor elementwise.
     */
    Tensor<T>& operator-=(const Tensor<T>& y) {
        binop(*this, y, *this, std::minus<T>());
        return *this;
    }

    /**
     * Subtracting the tensor with a scalar value.
     */
    Tensor<T> operator-(T v) const {
        Tensor<T> z(m_dims);
        scalarop(data(), v, z.data(), size(), std::minus<T>());
        return z;
    }

    /**
     * Inplace subtract a scalar value.
     */
    Tensor<T>& operator-=(T v) {
        scalarop(data(), v, data(), size(), std::minus<T>());
        return *this;
    }

    /**
     * Multiply two tensors elementwise.
     */
    Tensor<T> operator*(const Tensor<T>& y) const {
        Tensor<T> z(m_dims);
        binop(*this, y, z, std::multiplies<T>());
        return z;
    }

    /**
     * Inplace multiply another tensor elementwise.
     */
    Tensor<T>& operator*=(const Tensor<T>& y) {
        binop(*this, y, *this, std::multiplies<T>());
        return *this;
    }

    /**
     * Multiply the tensor with a scalar value.
     */
    Tensor<T> operator*(T v) const {
        Tensor<T> z(m_dims);
        scalarop(data(), v, z.data(), size(), std::multiplies<T>());
        return z;
    }

    /**
     * Inplace multiply a scalar value.
     */
    Tensor<T>& operator*=(T v) {
        scalarop(data(), v, data(), size(), std::multiplies<T>());
        return *this;
    }

    /**
     * Divides two tensors elementwise.
     */
    Tensor<T> operator/(const Tensor<T>& y) const {
        Tensor<T> z(m_dims);
        binop(*this, y, z, std::divides<T>());
        return z;
    }

    /**
     * Inplace divides another tensor elementwise.
     */
    Tensor<T>& operator/=(const Tensor<T>& y) {
        binop(*this, y, *this, std::divides<T>());
        return *this;
    }

    /**
     * Divides the tensor with a scalar value.
     */
    Tensor<T> operator/(T v) const {
        Tensor<T> z(m_dims);
        scalarop(data(), v, z.data(), size(), std::divides<T>());
        return z;
    }

    /**
     * Inplace divides a scalar value.
     */
    Tensor<T>& operator/=(T v) {
        scalarop(data(), v, data(), size(), std::divides<T>());
        return *this;
    }

    /**
     * Perform dot product on two matrices.
     */
    Tensor<T> dot(const Tensor<T>& y) const;

    /**
     * Transpose a matrix.
     */
    Tensor<T> transpose() const;

    /**
     * Apply a function on tensor's elements.
     */
    template <typename F>
    Tensor<T>& apply(F f) {
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
     * Casting element type.
     *
     * @tparam U the target element type
     * @return the Tensor with new element type.
     */
    template <typename U>
    Tensor<U> cast() const {
        return transform([](T x) { return static_cast<U>(x); });
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
        printRec(os, t.dims(), 0, t.data());
        return os;
    }

private:
    template <typename F>
    static void binop(const Tensor<T>& x, const Tensor<T>& y, Tensor<T>& z, F f) {
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
 * perform binary operation on two tensors elementwise.
 */
template <typename T, typename F>
inline Tensor<T> transform(const Tensor<T>& x, const Tensor<T>& y, F f) {
    assert(x.dims() == y.dims());
    Tensor<T> z(x.dims());
    std::transform(x.data(), x.data()+x.size(), y.data(), z.data(), f);
    return z;
}

template <typename T>
Tensor<T> Tensor<T>::dot(const Tensor<T>& y) const {
    assert(dims().size() == 2 && y.dims().size() == 2);

    size_t n = m_dims[0];
    size_t p = m_dims[1];
    size_t m = y.m_dims[1];
    assert(p == y.m_dims[0]);

    Tensor<T> z({n, m});
    const T* px = data();
    const T* py = y.data();
    T* pz = z.data();
    int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            T v = 0;
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

    size_t n = m_dims[0];
    size_t m = m_dims[1];

    Tensor<T> y({m, n});
    const T* px = data();
    T* py = y.data();
    int i, j;

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            *py++ = px[j * m + i];
    return y;
}

template <typename T>
const T* Tensor<T>::printRec(std::ostream& out, const std::vector<size_t>& dims, size_t level, const T* data) {
    size_t d = dims[level];

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
