#pragma once

#include "tensor/shape.h"
#include "tensor/host.h"
#include "gpgpu.h"
#include "gblas.h"
#include "gdnn.h"

namespace dlf {

/**
 * A tensor which data allocated from compute devices such as GPU.
 */
template <typename T>
class DevTensor : public Shaped {
    gpgpu::Buffer<T> m_data;

public:
    DevTensor() = default;

    DevTensor(Shape shape, const gpgpu::Queue& queue = gpgpu::current::queue())
        : Shaped(std::move(shape))
    {
        m_data = queue.context().template createBuffer<T>(size());
    }

    DevTensor(const Tensor<T>& host, const gpgpu::Queue& queue = gpgpu::current::queue())
        : Shaped(host.shape())
    {
        m_data = queue.context().template createBuffer<T>(size());
        m_data.write(queue, host.data(), host.size());
    }

    DevTensor(Shape shape, gpgpu::Buffer<T> data)
        : Shaped(std::move(shape)), m_data(std::move(data))
    {
    }

    /**
     * Read data from device.
     */
    Tensor<T> read(const gpgpu::Queue& queue = gpgpu::current::queue()) const {
        Tensor<T> host(shape());
        m_data.read(queue, host.data(), host.size());
        return host;
    }

    /**
     * Asynchronously read data from device.
     */
    Tensor<T> readAsync(const gpgpu::Queue& queue = gpgpu::current::queue()) const {
        Tensor<T> host(shape());
        m_data.readAsync(queue, host.data(), host.size());
        return host;
    };

    /**
     * Read data from device and store data into given host tensor.
     */
    void read(Tensor<T>& host, const gpgpu::Queue& queue = gpgpu::current::queue()) const {
        assert(shape() == host.shape());
        m_data.read(queue, host.data(), host.size());
    }

    /**
     * Asynchronously read data from device and store data into given host tensor.
     */
    void readAsync(Tensor<T>& host, const gpgpu::Queue& queue = gpgpu::current::queue()) const {
        assert(shape() == host.shape());
        m_data.readAsync(queue, host.data(), host.size());
    }

    /**
     * Write data into device.
     */
    void write(const Tensor<T>& host, const gpgpu::Queue& queue = gpgpu::current::queue()) {
        assert(shape() == host.shape());
        m_data.write(queue, host.data(), host.size());
    }

    /**
     * Asynchronously write data into device.
     */
    void writeAsync(const Tensor<T>& host, const gpgpu::Queue& queue = gpgpu::current::queue()) {
        assert(shape() == host.shape());
        m_data.writeAsync(queue, host.data(), host.size());
    }

    /**
     * Copy data into target.
     */
    void copyTo(DevTensor<T>& dest, const gpgpu::Queue& queue = gpgpu::current::queue()) const {
        assert(shape() == dest.shape());
        m_data.copyTo(queue, dest.data(), size());
    }

    /**
     * Asynchronously copy data into destination.
     */
    void copyToAsync(DevTensor<T>& dest, const gpgpu::Queue& queue = gpgpu::current::queue()) const {
        assert(shape() == dest.shape());
        m_data.copyToAsync(queue, dest.data(), size());
    }

    /**
     * Create a copy of this tensor.
     */
    DevTensor<T> copy(const gpgpu::Queue& queue = gpgpu::current::queue()) const {
        DevTensor<T> dest(shape(), queue);
        m_data.copyTo(queue, dest.data(), size());
        return dest;
    }

    /**
     * Asynchronously create a copy of this tensor.
     */
    DevTensor<T> copyAsync(const gpgpu::Queue& queue = gpgpu::current::queue()) const {
        DevTensor<T> dest(shape(), queue);
        m_data.copyToAsync(queue, dest.data(), size());
        return dest;
    }

    /**
     * Get the underlying device buffer.
     */
    gpgpu::Buffer<T>& data() noexcept {
        return m_data;
    }

    /**
     * Get the underlying device buffer.
     */
    const gpgpu::Buffer<T>& data() const noexcept {
        return m_data;
    }

public:
    DevTensor<T>& operator+=(const DevTensor<T>& rhs);
    DevTensor<T>& operator-=(const DevTensor<T>& rhs);
    DevTensor<T>& operator*=(const DevTensor<T>& rhs);
    DevTensor<T>& operator*=(const T& rhs);
    DevTensor<T>& operator/=(const DevTensor<T>& rhs);
};

//==-------------------------------------------------------------------------
// DevTensor unary transformations
//==-------------------------------------------------------------------------

template <typename T>
inline DevTensor<T> abs(const DevTensor<T>& x, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    DevTensor<T> y(x.shape(), queue);
    gpgpu::dnn::abs(x.size(), x.data(), y.data(), queue);
    return y;
}

template <typename T>
inline DevTensor<T> abs(DevTensor<T>&& x, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    gpgpu::dnn::abs(x.size(), x.data(), x.data(), queue);
    return std::move(x);
}

template <typename T>
inline void abs(const DevTensor<T>& x, DevTensor<T>& y, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    assert(x.shape() == y.shape());
    gpgpu::dnn::abs(x.size(), x.data(), y.data(), queue);
}

template <typename T>
inline DevTensor<T> neg(const DevTensor<T>& x, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    DevTensor<T> y(x.shape(), queue);
    gpgpu::dnn::neg(x.size(), x.data(), y.data(), queue);
    return y;
}

template <typename T>
inline DevTensor<T> neg(DevTensor<T>&& x, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    gpgpu::dnn::neg(x.size(), x.data(), x.data(), queue);
    return std::move(x);
}

template <typename T>
inline void neg(const DevTensor<T>& x, DevTensor<T>& y, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    assert(x.shape() == y.shape());
    gpgpu::dnn::neg(x.size(), x.data(), y.data(), queue);
}

template <typename T>
inline DevTensor<T> sign(const DevTensor<T>& x, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    DevTensor<T> y(x.shape(), queue);
    gpgpu::dnn::sign(x.size(), x.data(), y.data(), queue);
    return y;
}

template <typename T>
inline DevTensor<T> sign(DevTensor<T>&& x, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    gpgpu::dnn::sign(x.size(), x.data(), x.data(), queue);
    return std::move(x);
}


template <typename T>
inline void sign(const DevTensor<T>& x, DevTensor<T>& y, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    assert(x.shape() == y.shape());
    gpgpu::dnn::sign(x.size(), x.data(), y.data(), queue);
}

#define DEFINE_TRANSFORM(name) \
template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>> \
inline DevTensor<T> name(const DevTensor<T>& x, const gpgpu::Queue& queue = gpgpu::current::queue()) { \
    DevTensor<T> y(x.shape(), queue); \
    gpgpu::dnn::transform(#name, x.size(), x.data(), y.data(), queue); \
    return y; \
} \
template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>> \
inline DevTensor<T> name(DevTensor<T>&& x, const gpgpu::Queue& queue = gpgpu::current::queue()) { \
    gpgpu::dnn::transform(#name, x.size(), x.data(), x.data(), queue); \
    return std::move(x); \
} \
template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>> \
inline void name(const DevTensor<T>& x, DevTensor<T>& y, const gpgpu::Queue& queue = gpgpu::current::queue()) { \
    assert(x.shape() == y.shape()); \
    gpgpu::dnn::transform(#name, x.size(), x.data(), y.data(), queue); \
}


DEFINE_TRANSFORM(reciprocal)
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
// DevTensor binary transformations
//==-------------------------------------------------------------------------

#define DEFINE_BINARY(name) \
template <typename T> \
inline DevTensor<T> name(const DevTensor<T>& x, const DevTensor<T>& y, \
                         const gpgpu::Queue& queue = gpgpu::current::queue()) { \
    assert(x.shape() == y.shape()); \
    DevTensor<T> z(x.shape(), queue); \
    gpgpu::dnn::name(x.size(), x.data(), y.data(), z.data(), queue); \
    return z; \
} \
template <typename T> \
inline DevTensor<T> name(DevTensor<T>&& x, const DevTensor<T>& y, \
                         const gpgpu::Queue& queue = gpgpu::current::queue()) { \
    assert(x.shape() == y.shape()); \
    gpgpu::dnn::name(x.size(), x.data(), y.data(), x.data(), queue); \
    return std::move(x); \
} \
template <typename T> \
inline DevTensor<T> name(const DevTensor<T>& x, DevTensor<T>&& y, \
                         const gpgpu::Queue& queue = gpgpu::current::queue()) { \
    assert(x.shape() == y.shape()); \
    gpgpu::dnn::name(x.size(), x.data(), y.data(), y.data(), queue); \
    return std::move(y); \
} \
template <typename T> \
inline DevTensor<T> name(DevTensor<T>&& x, DevTensor<T>&& y, \
                         const gpgpu::Queue& queue = gpgpu::current::queue()) { \
    assert(x.shape() == y.shape()); \
    gpgpu::dnn::name(x.size(), x.data(), y.data(), x.data(), queue); \
    return std::move(x); \
} \
template <typename T> \
inline void name##To(const DevTensor<T>& x, const DevTensor<T>& y, DevTensor<T>& z, \
                     const gpgpu::Queue& queue = gpgpu::current::queue()) { \
    assert(x.shape() == y.shape() && x.shape() == z.shape()); \
    gpgpu::dnn::name(x.size(), x.data(), y.data(), z.data(), queue); \
}

#define DEFINE_BINARY_OP(name, op) \
DEFINE_BINARY(name) \
template <typename T> \
inline DevTensor<T>& DevTensor<T>::operator op##=(const DevTensor<T>& rhs) { \
    name##To(*this, rhs, *this); \
    return *this; \
} \
template <typename T> \
inline DevTensor<T> operator op(const DevTensor<T>& lhs, const DevTensor<T>& rhs) { \
    return name(lhs, rhs); \
} \
template <typename T> \
inline DevTensor<T> operator op(DevTensor<T>&& lhs, const DevTensor<T>& rhs) { \
    return name(std::move(lhs), rhs); \
} \
template <typename T> \
inline DevTensor<T> operator op(const DevTensor<T>& lhs, DevTensor<T>&& rhs) { \
    return name(lhs, std::move(rhs)); \
} \
template <typename T> \
inline DevTensor<T> operator op(DevTensor<T>&& lhs, DevTensor<T>&& rhs) { \
    return name(std::move(lhs), std::move(rhs)); \
}

DEFINE_BINARY_OP(add, +)
DEFINE_BINARY_OP(sub, -)
DEFINE_BINARY_OP(mul, *)
DEFINE_BINARY_OP(div, /)
DEFINE_BINARY(pow)

#undef DEFINE_BINARY_OP
#undef DEFINE_BINARY

template <typename T>
inline DevTensor<T>& DevTensor<T>::operator*=(const T& rhs) {
    gblas::scal(size(), rhs, data(), 1);
    return *this;
}

template <typename T>
inline DevTensor<T> operator*(const DevTensor<T>& lhs, const T& rhs) {
    auto R = lhs.copyAsync();
    R *= rhs;
    return R;
}

template <typename T>
inline DevTensor<T> operator*(DevTensor<T>&& lhs, const T& rhs) {
    lhs *= rhs;
    return std::move(lhs);
}

template <typename T>
inline DevTensor<T> operator*(const T& lhs, const DevTensor<T>& rhs) {
    auto R = rhs.copyAsync();
    R *= lhs;
    return R;
}

template <typename T>
inline DevTensor<T> operator*(const T& lhs, DevTensor<T>&& rhs) {
    rhs *= lhs;
    return std::move(rhs);
}

template <typename T>
inline DevTensor<T> operator-(const DevTensor<T>& x) {
    return neg(x);
}

template <typename T>
inline DevTensor<T> operator-(DevTensor<T>&& x) {
    return neg(std::move(x));
}

//==-------------------------------------------------------------------------
// DevTensor production
//==-------------------------------------------------------------------------

/**
 * Perform dot product on two tensors. The tensors must be vector
 * or matrix and have compatible dimensions.
 */
template <typename T>
DevTensor<T>& dot(const DevTensor<T>& A, const DevTensor<T>& B, DevTensor<T>* C,
                  const gpgpu::Queue& queue = gpgpu::current::queue(),
                  gpgpu::Event* event = nullptr)
{
    if (A.is_vector() && B.is_vector()) {
        auto n = A.extent(0);
        assert(n == B.extent(0));
        assert(C->is_vector() && 1 == C->extent(0));
        gblas::dot(n, A.data(), 1, B.data(), 1, C->data(), queue, event);
    } else if (A.is_matrix() && B.is_vector()) {
        auto m = A.extent(0), n = A.extent(1);
        assert(n == B.extent(0));
        assert(C->is_vector() && m == C->extent(0));
        gblas::gemv(gblas::Layout::RowMajor,
                    gblas::Transpose::NoTrans,
                    m, n, T(1),
                    A.data(), A.stride(0),
                    B.data(), 1, T(0),
                    C->data(), 1,
                    queue, event);
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

        gblas::gemm(gblas::Layout::RowMajor,
                    gblas::Transpose::NoTrans,
                    gblas::Transpose::NoTrans,
                    C_shape.extent(0), C_shape.extent(1), A_shape.extent(1),
                    T(1),
                    A.data(), A_shape.stride(0),
                    B.data(), B_shape.stride(0),
                    T(0),
                    C->data(), C_shape.stride(0),
                    queue, event);
    } else {
        assert(false);
    }
    return *C;
}

template <typename T>
DevTensor<T> dot(const DevTensor<T>& A, const DevTensor<T>& B,
                 const gpgpu::Queue& queue = gpgpu::current::queue(),
                 gpgpu::Event* event = nullptr)
{
    if (A.is_vector() && B.is_vector()) {
        assert(A.shape() == B.shape());
        DevTensor<T> C({1}, queue);
        dot(A, B, &C, queue, event);
        return C;
    } else if (A.is_matrix() && B.is_vector()) {
        assert(A.extent(1) == B.extent(0));
        DevTensor<T> C({A.extent(0)}, queue);
        dot(A, B, &C, queue, event);
        return C;
    } else if (A.is_vector() && B.is_matrix()) {
        assert(A.extent(0) == B.extent(0));
        DevTensor<T> C({B.extent(1)}, queue);
        dot(A, B, &C, queue, event);
        return C;
    } else if (A.is_matrix() && B.is_matrix()) {
        auto m = A.extent(0), k = A.extent(1);
        auto p = B.extent(0), n = B.extent(1);
        assert(k == p);
        DevTensor<T> C({m, n}, queue);
        dot(A, B, &C, queue, event);
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
void gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
          const T& beta, DevTensor<T>* C,
          bool transA = false, bool transB = false,
          const gpgpu::Queue& queue = gpgpu::current::queue(),
          gpgpu::Event* event = nullptr)
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

    gblas::gemm(gblas::Layout::RowMajor,
                transA ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
                transB ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
                m, n, k,
                alpha,
                A.data(), A.stride(0),
                B.data(), B.stride(0),
                beta,
                C->data(), C->stride(0),
                queue, event);
}

template <typename T>
DevTensor<T> gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
                  const T& beta, DevTensor<T>& C,
                  bool transA = false, bool transB = false,
                  const gpgpu::Queue& queue = gpgpu::current::queue(),
                  gpgpu::Event* event = nullptr) {
    DevTensor<T> R = C.copy(queue);
    gemm(alpha, A, B, beta, &R, transA, transB, queue, event);
    return R;
}

} // namespace dlf
