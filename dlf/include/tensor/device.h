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

    explicit DevTensor(Shape shape) : Shaped(std::move(shape)){
        m_data = gpgpu::current::context().createBuffer<T>(size());
    }

    DevTensor(const Tensor<T>& host) : Shaped(host.shape()) {
        m_data = gpgpu::current::context().createBuffer<T>(size());
        m_data.write(gpgpu::current::queue(), host.data(), host.size());
    }

    DevTensor(Shape shape, gpgpu::Buffer<T> data)
        : Shaped(std::move(shape)), m_data(std::move(data))
    {}

    DevTensor(const DevTensor& src) : DevTensor(src.shape()) {
        src.copyTo(*this);
    }

    DevTensor& operator=(const DevTensor& src) {
        if (size() != src.size() || m_data.handle() == nullptr)
            m_data = gpgpu::current::context().createBuffer<T>(src.size());
        Shaped::operator=(src);
        src.copyTo(*this);
        return *this;
    }

    DevTensor(DevTensor&&) = default;
    DevTensor& operator=(DevTensor&&) = default;

    /**
     * Read data from device.
     */
    Tensor<T> read() const {
        Tensor<T> host(shape());
        m_data.read(gpgpu::current::queue(), host.data(), host.size());
        return host;
    }

    /**
     * Read data from device and store data into given host tensor.
     */
    void readTo(Tensor<T>& host) const {
        assert(shape() == host.shape());
        m_data.read(gpgpu::current::queue(), host.data(), host.size());
    }

    /**
     * Asynchronously read data from device and store data into given host tensor.
     */
    void readToAsync(Tensor<T>& host) const {
        assert(shape() == host.shape());
        m_data.readAsync(gpgpu::current::queue(), host.data(), host.size());
    }

    /**
     * Write data into device.
     */
    void write(const Tensor<T>& host) {
        assert(shape() == host.shape());
        m_data.write(gpgpu::current::queue(), host.data(), host.size());
    }

    /**
     * Asynchronously write data into device.
     */
    void writeAsync(const Tensor<T>& host) {
        assert(shape() == host.shape());
        m_data.writeAsync(gpgpu::current::queue(), host.data(), host.size());
    }

    /**
     * Copy data into target.
     */
    void copyTo(DevTensor<T>& dest) const {
        if (m_data != dest.data()) {
            m_data.copyTo(gpgpu::current::queue(), dest.data(), size());
        }
    }

    /**
     * Asynchronously copy data into destination.
     */
    void copyToAsync(DevTensor<T>& dest) const {
        if (m_data != dest.data()) {
            m_data.copyToAsync(gpgpu::current::queue(), dest.data(), size());
        }
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
     * Create a scalar.
     */
    static DevTensor scalar(const T& value) {
        DevTensor<T> res({1});
        res.data().write(gpgpu::current::queue(), &value, 1);
        return res;
    }
};

template <typename T>
inline DevTensor<T> dev(const Tensor<T>& host) {
    return DevTensor<T>(host);
}

//==-------------------------------------------------------------------------
// DevTensor unary transformations
//==-------------------------------------------------------------------------

template <typename T, typename Fn>
std::enable_if_t<!std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensor<T>&>
inline transformTo(const DevTensor<T>& x, DevTensor<T>& y, Fn) {
    assert(x.shape() == y.shape());
    gpgpu::dnn::transform(Fn::name, x.size(), x.data(), y.data());
    return y;
}

template <typename T, typename Fn>
std::enable_if_t<std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensor<T>&>
inline transformTo(const DevTensor<T>& X, DevTensor<T>& Y, Fn fn) {
    assert(X.shape() == Y.shape());
    gpgpu::dnn::transform(Fn::name, X.size(), fn.alpha, fn.beta, X.data(), Y.data());
    return Y;
}

template <typename T, typename Fn>
inline DevTensor<T> transform(const DevTensor<T>& x, Fn fn) {
    DevTensor<T> y(x.shape());
    transformTo(x, y, fn);
    return y;
}

template <typename T, typename Fn>
inline DevTensor<T> transform(DevTensor<T>&& x, Fn fn) {
    return std::move(transformTo(x, x, fn));
}

template <typename T>
void reorder(const DevTensor<T>& src, const Shape& shape, DevTensor<T>& dst) {
    assert(dst.shape() == shape);
    if (src.data() == dst.data())
        return;
    if (src.size() == shape.size() && shape.is_contiguous()) {
        src.copyToAsync(dst);
    } else if (src.shape().is_tail(shape)) {
        gpgpu::dnn::copy(src.size(), src.data(), shape.offset(),
                         dst.size(), dst.data(), 0);
    } else {
        gpgpu::dnn::copy(shape.size(),
                         src.data(), shape.offset(),
                         dst.data(), 0,
                         shape.strides(), shape.extents());
    }
}

template <typename T>
inline void flat_copy(const DevTensor<T>& src, DevTensor<T>& dst) {
    assert(src.size() == dst.size());
    src.copyToAsync(dst);
}

//==-------------------------------------------------------------------------
// DevTensor binary transformations
//==-------------------------------------------------------------------------

template <typename T, typename Fn>
DevTensor<T>& transformTo(const DevTensor<T>& A, const DevTensor<T>& B, DevTensor<T>& C, Fn fn) {
    Shape final_shape = Shape::broadcast(A, B);
    if (C.shape() != final_shape) {
        throw shape_error("incompatible shape");
    }

    if (A.shape().is_tail(B.shape()) || B.shape().is_tail(A.shape())) {
        gpgpu::dnn::transform(Fn::name, A.size(), A.data(), B.size(), B.data(), C.data());
        return C;
    }

    int axis = A.shape().pole(B.shape());
    if (axis != -1) {
        transformChannel(A, B, C, axis, fn);
        return C;
    }

    auto shape_A = A.shape().broadcast(final_shape);
    auto shape_B = B.shape().broadcast(final_shape);
    gpgpu::dnn::transform(Fn::name, final_shape.size(), A.data(), B.data(), C.data(),
                          shape_A.strides(), shape_B.strides(), final_shape.extents());
    return C;
}

template <typename T, typename Fn>
void transformChannel(const DevTensor<T>& A, const DevTensor<T>& B, DevTensor<T>& C, size_t axis, Fn) {
    assert(B.is_vector() || A.shape().pole(B.shape()) == axis);
    assert(axis < A.rank());
    assert(A.extent(axis) == B.size());
    assert(C.shape() == A.shape());

    size_t m = 1;
    for (int i = 0; i <= axis; i++)
        m *= A.extent(i);
    size_t n = A.size() / m;

    gpgpu::dnn::transform(Fn::name, m, n, B.size(), A.data(), B.data(), C.data());
}

template <typename T, typename Fn>
inline DevTensor<T> transform(const DevTensor<T>& A, const DevTensor<T>& B, Fn fn) {
    DevTensor<T> C(Shape::broadcast(A, B));
    transformTo(A, B, C, fn);
    return C;
}

template <typename T, typename Fn>
inline DevTensor<T> transform(DevTensor<T>&& A, const DevTensor<T>& B, Fn fn) {
    if (A.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, A, fn));
    else
        return transform(A, B, fn);
}

template <typename T, typename Fn>
inline DevTensor<T> transform(const DevTensor<T>& A, DevTensor<T>&& B, Fn fn) {
    if (B.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, B, fn));
    else
        return transform(A, B, fn);
}

template <typename T, typename Fn>
inline DevTensor<T> transform(DevTensor<T>&& A, DevTensor<T>&& B, Fn fn) {
    Shape final_shape = Shape::broadcast(A, B);
    if (A.shape() == final_shape)
        return std::move(transformTo(A, B, A, fn));
    else if (B.shape() == final_shape)
        return std::move(transformTo(A, B, B, fn));
    else
        return transform(A, B, fn);
}

//==-------------------------------------------------------------------------
// DevTensor production
//==-------------------------------------------------------------------------

/**
 * Perform dot product on two tensors. The tensors must be vector
 * or matrix and have compatible dimensions.
 */
template <typename T>
DevTensor<T>& dot(const DevTensor<T>& A, const DevTensor<T>& B, DevTensor<T>* C) {
    if (A.is_vector() && B.is_vector()) {
        auto n = A.extent(0);
        assert(n == B.extent(0));
        assert(C->is_scalar());
        gblas::dot(n, A.data(), 1, B.data(), 1, C->data());
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
                    C->data(), 1);
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
                    C->data(), C_shape.stride(0));
        return *C;
    }

    throw std::logic_error("dot: unsupported tensor shape");
}

/**
 * General matrix multiplication.
 */
template <typename T>
void gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
          const T& beta, DevTensor<T>* C,
          bool transA = false, bool transB = false,
          DevTensor<T>* work = nullptr)
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
                work == nullptr ? nullptr : &work->data());
}

template <typename T>
inline size_t gemmWorkspaceSize(
    const DevTensor<T>& A, const DevTensor<T>& B, const DevTensor<T>& C,
    bool transA = false, bool transB = false)
{
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    if (k != p || m != C.extent(0) || n != C.extent(1))
        throw shape_error("gemm: incompatible shape");

    return gblas::gemmTempBufferSize<T>(
        gblas::Layout::RowMajor,
        transA ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
        transB ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
        m, n, k, 0, A.stride(0), 0, B.stride(0), 0, C.stride(0));
}

} // namespace dlf
