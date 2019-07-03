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
    static constexpr bool is_tensor = true;
    using value_type = T;

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

    /**
     * Broadcast this tensor to the given shape.
     */
    DevTensor<T> broadcast(const Shape& shape, const gpgpu::Queue& queue = gpgpu::current::queue()) const &;
    DevTensor<T> broadcast(const Shape& shape, const gpgpu::Queue& queue = gpgpu::current::queue()) &&;

public:
    DevTensor<T>& operator+=(const DevTensor<T>& rhs);
    DevTensor<T>& operator-=(const DevTensor<T>& rhs);
    DevTensor<T>& operator*=(const DevTensor<T>& rhs);
    DevTensor<T>& operator/=(const DevTensor<T>& rhs);
};

template <typename T>
inline DevTensor<T> dev(const Tensor<T>& host, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    return DevTensor<T>(host, queue);
}

template <typename T>
inline DevTensor<T> dev(T value, const gpgpu::Queue& queue = gpgpu::current::queue()) {
    DevTensor<T> res({1}, queue);
    res.data().write(queue, &value, 1);
    return res;
}

//==-------------------------------------------------------------------------
// DevTensor unary transformations
//==-------------------------------------------------------------------------

template <typename T, typename Op>
std::enable_if_t<!std::is_base_of<xfn::parameterized_function<T>, Op>::value, DevTensor<T>&>
inline transformTo(const DevTensor<T>& x, DevTensor<T>& y, Op,
                   const gpgpu::Queue& queue = gpgpu::current::queue())
{
    assert(x.shape() == y.shape());
    gpgpu::dnn::transform(Op::name, x.size(), x.data(), y.data(), queue);
    return y;
}

template <typename T, typename Op>
std::enable_if_t<std::is_base_of<xfn::parameterized_function<T>, Op>::value, DevTensor<T>&>
inline transformTo(const DevTensor<T>& X, DevTensor<T>& Y, Op op,
                   const gpgpu::Queue& queue = gpgpu::current::queue())
{
    assert(X.shape() == Y.shape());
    gpgpu::dnn::transform(Op::name, X.size(), op.alpha, op.beta, X.data(), Y.data(), queue);
    return Y;
}

template <typename T, typename Op>
inline DevTensor<T> transform(const DevTensor<T>& x, Op&& op,
    const gpgpu::Queue& queue = gpgpu::current::queue())
{
    DevTensor<T> y(x.shape(), queue);
    transformTo(x, y, std::forward<Op>(op), queue);
    return y;
}

template <typename T, typename Op>
inline DevTensor<T> transform(DevTensor<T>&& x, Op&& op,
    const gpgpu::Queue& queue = gpgpu::current::queue())
{
    return std::move(transformTo(x, x, std::forward<Op>(op), queue));
}

template <typename T>
inline void copy(const DevTensor<T>& src, DevTensor<T>& dst,
                 const gpgpu::Queue& queue = gpgpu::current::queue(),
                 gpgpu::Event* event = nullptr)
{
    if (&src != &dst) {
        if (src.shape() == dst.shape()) {
            src.copyToAsync(dst, queue);
        } else if (src.shape().is_tail(dst.shape())) {
            gpgpu::dnn::copy(src.size(), src.data(), dst.size(), dst.data(), queue, event);
        } else {
            gpgpu::dnn::copy(dst.size(), src.data(), dst.data(),
                             src.shape().broadcast(dst.shape()).strides(),
                             dst.shape().extents(), queue, event);
        }
    }
}

template <typename T>
inline DevTensor<T> DevTensor<T>::broadcast(const Shape& shape, const gpgpu::Queue& queue) const & {
    DevTensor<T> dst(shape, queue);
    dlf::copy(*this, dst, queue);
    return dst;
}

template <typename T>
inline DevTensor<T> DevTensor<T>::broadcast(const Shape& shape, const gpgpu::Queue& queue) && {
    if (shape == this->shape()) {
        return std::move(*this);
    } else {
        DevTensor<T> dst(shape, queue);
        dlf::copy(*this, dst, queue);
        return dst;
    }
}

//==-------------------------------------------------------------------------
// DevTensor binary transformations
//==-------------------------------------------------------------------------

template <typename T, typename Op>
DevTensor<T>& transformTo(const DevTensor<T>& x, const DevTensor<T>& y, DevTensor<T>& z, Op,
                          const gpgpu::Queue& queue = gpgpu::current::queue())
{
    if (x.shape().is_tail(y.shape()) || y.shape().is_tail(x.shape())) {
        assert(z.shape() == Shape::broadcast(x, y));
        gpgpu::dnn::transform(Op::name, x.size(), x.data(), y.size(), y.data(), z.data(), queue);
    } else {
        Shape final_shape = Shape::broadcast(x, y);
        gpgpu::dnn::transform(Op::name, final_shape.size(), x.data(), y.data(), z.data(),
                              x.shape().broadcast(final_shape).strides(),
                              y.shape().broadcast(final_shape).strides(),
                              final_shape.extents());
    }
    return z;
}

template <typename T, typename Op>
inline DevTensor<T> transform(const DevTensor<T>& x, const DevTensor<T>& y, Op&& op,
                              const gpgpu::Queue& queue = gpgpu::current::queue())
{
    DevTensor<T> z(Shape::broadcast(x, y));
    transformTo(x, y, z, std::forward<Op>(op), queue);
    return z;
}

template <typename T, typename Op>
inline DevTensor<T> transform(DevTensor<T>&& x, const DevTensor<T>& y, Op&& op,
                              const gpgpu::Queue& queue = gpgpu::current::queue())
{
    if (x.shape() == Shape::broadcast(x, y))
        return std::move(transformTo(x, y, x, std::forward<Op>(op), queue));
    else
        return transform(x, y, std::forward<Op>(op), queue);
}

template <typename T, typename Op>
inline DevTensor<T> transform(const DevTensor<T>& x, DevTensor<T>&& y, Op&& op,
                              const gpgpu::Queue& queue = gpgpu::current::queue())
{
    if (y.shape() == Shape::broadcast(x, y))
        return std::move(transformTo(x, y, y, std::forward<Op>(op), queue));
    else
        return transform(x, y, std::forward<Op>(op), queue);
}

template <typename T, typename Op>
inline DevTensor<T> transform(DevTensor<T>&& x, DevTensor<T>&& y, Op&& op,
                              const gpgpu::Queue& queue = gpgpu::current::queue())
{
    Shape final_shape = Shape::broadcast(x, y);
    if (x.shape() == final_shape)
        return std::move(transformTo(x, y, x, std::forward<Op>(op), queue));
    else if (y.shape() == final_shape)
        return std::move(transformTo(x, y, y, std::forward<Op>(op), queue));
    else
        return transform(x, y, std::forward<Op>(op), queue);
}

#define DEFINE_BINARY_OP(op, fn) \
template <typename T> \
inline DevTensor<T>& DevTensor<T>::operator op##=(const DevTensor<T>& rhs) { \
    return transformTo(*this, rhs, *this, dlf::xfn::fn<>()); \
} \
template <typename T> \
inline DevTensor<T> operator op(const DevTensor<T>& lhs, const DevTensor<T>& rhs) { \
    return transform(lhs, rhs, dlf::xfn::fn<>()); \
} \
template <typename T> \
inline DevTensor<T> operator op(DevTensor<T>&& lhs, const DevTensor<T>& rhs) { \
    return transform(std::move(lhs), rhs, dlf::xfn::fn<>()); \
} \
template <typename T> \
inline DevTensor<T> operator op(const DevTensor<T>& lhs, DevTensor<T>&& rhs) { \
    return transform(lhs, std::move(rhs), dlf::xfn::fn<>()); \
} \
template <typename T> \
inline DevTensor<T> operator op(DevTensor<T>&& lhs, DevTensor<T>&& rhs) { \
    return transform(std::move(lhs), std::move(rhs), dlf::xfn::fn<>()); \
}

DEFINE_BINARY_OP(+, plus)
DEFINE_BINARY_OP(-, minus)
DEFINE_BINARY_OP(*, multiplies)
DEFINE_BINARY_OP(/, divides)
#undef DEFINE_BINARY_OP

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
        assert(C->is_scalar());
        gblas::dot(n, A.data(), 1, B.data(), 1, C->data(), queue, event);
        return *C;
    }

    if (A.is_matrix() && B.is_vector()) {
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
        return *C;
    }

    assert(false);
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
    if (k != p || m != C->extent(0) || n != C->extent(1))
        throw shape_error("gemm: incompatible shape");

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
void gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
          const T& beta, const DevTensor<T>& C, DevTensor<T>& Y,
          bool transA = false, bool transB = false,
          const gpgpu::Queue& queue = gpgpu::current::queue(),
          gpgpu::Event* event = nullptr)
{
    copy(C, Y, queue);
    gemm(alpha, A, B, beta, &Y, transA, transB, queue, event);
}

template <typename T>
inline DevTensor<T> gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
                         const T& beta, const DevTensor<T>& C,
                         bool transA = false, bool transB = false,
                         const gpgpu::Queue& queue = gpgpu::current::queue(),
                         gpgpu::Event* event = nullptr)
{
    assert(A.is_matrix() && B.is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);

    auto Y = C.broadcast({m, n}, queue);
    gemm(alpha, A, B, beta, &Y, transA, transB, queue, event);
    return Y;
}

} // namespace dlf
