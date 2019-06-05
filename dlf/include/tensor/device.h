#pragma once

#include "tensor/shape.h"
#include "tensor/host.h"
#include "gpgpu.h"
#include "gpblas.h"

namespace dlf {

/**
 * A tensor which data allocated from compute devices such as GPU.
 */
template <typename T>
class DevTensor : public Shaped {
    gpgpu::Queue m_queue;
    gpgpu::Buffer<T> m_data;

public:
    DevTensor() = default;

    DevTensor(Shape shape, gpgpu::Queue queue)
        : Shaped(std::move(shape)), m_queue(std::move(queue))
    {
        m_data = m_queue.context().template createBuffer<T>(size());
    }

    DevTensor(const Tensor<T>& host, gpgpu::Queue queue)
        : Shaped(host.shape()), m_queue(std::move(queue))
    {
        m_data = m_queue.context().template createBuffer<T>(size());
        m_data.write(m_queue, host.data(), host.size());
    }

    DevTensor(Shape shape, gpgpu::Buffer<T> data, gpgpu::Queue queue)
        : Shaped(std::move(shape)), m_data(std::move(data)), m_queue(std::move(queue))
    {
    }

    /**
     * Read data from device.
     */
    Tensor<T> read() const {
        Tensor<T> host(shape());
        m_data.read(m_queue, host.data(), host.size());
        return host;
    }

    /**
     * Asynchronously read data from device.
     */
    Tensor<T> readAsync() const {
        Tensor<T> host(shape());
        m_data.readAsync(m_queue, host.data(), host.size());
        return host;
    };

    /**
     * Read data from device and store data into given host tensor.
     */
    void read(Tensor<T>& host) const {
        assert(shape() == host.shape());
        m_data.read(m_queue, host.data(), host.size());
    }

    /**
     * Asynchronously read data from device and store data into given host tensor.
     */
    void readAsync(Tensor<T>& host) const {
        assert(shape() == host.shape());
        m_data.readAsync(m_queue, host.data(), host.size());
    }

    /**
     * Write data into device.
     */
    void write(const Tensor<T>& host) {
        assert(shape() == host.shape());
        m_data.write(m_queue, host.data(), host.size());
    }

    /**
     * Asynchronously write data into device.
     */
    void writeAsync(const Tensor<T>& host) {
        assert(shape() == host.shape());
        m_data.writeAsync(m_queue, host.data(), host.size());
    }

    /**
     * Copy data into target.
     */
    void copyTo(DevTensor<T>& dest) const {
        assert(shape() == dest.shape());
        m_data.copyTo(m_queue, dest.buffer(), size());
    }

    /**
     * Asynchronously copy data into destination.
     */
    void copyToAsync(DevTensor<T>& dest) const {
        assert(shape() == dest.shape());
        m_data.copyToAsync(m_queue, dest.buffer(), size());
    }

    /**
     * Create a copy of this tensor.
     */
    DevTensor<T> copy() const {
        DevTensor<T> dest(shape(), m_queue);
        m_data.copyTo(m_queue, dest.buffer(), size());
        return dest;
    }

    /**
     * Asynchronously create a copy of this tensor.
     */
    DevTensor<T> copyAsync() const {
        DevTensor<T> dest(shape(), m_queue);
        m_data.copyToAsync(m_queue, dest.buffer(), size());
        return dest;
    }

    /**
     * Get the underlying device buffer.
     */
    gpgpu::Buffer<T>& buffer() noexcept {
        return m_data;
    }

    /**
     * Get the underlying device buffer.
     */
    const gpgpu::Buffer<T>& buffer() const noexcept {
        return m_data;
    }

    /**
     * Returns the queue associated with this device tensor.
     */
    gpgpu::Queue& queue() noexcept {
        return m_queue;
    }

    /**
     * Returns the queue associated with this device tensor.
     */
    const gpgpu::Queue& queue() const noexcept {
        return m_queue;
    }

public:
    DevTensor<T>& operator+=(const DevTensor<T>& rhs);
    DevTensor<T>& operator-=(const DevTensor<T>& rhs);
    DevTensor<T>& operator*=(const DevTensor<T>& rhs);
    DevTensor<T>& operator*=(const T& rhs);
};

template <typename T>
inline DevTensor<T>& DevTensor<T>::operator+=(const DevTensor<T>& rhs) {
    assert(shape() == rhs.shape());
    gpgpu::blas::axpy(size(), T(1), rhs.buffer(), 0, 1, buffer(), 0, 1, m_queue);
    return *this;
}

template <typename T>
inline DevTensor<T> operator+(const DevTensor<T>& lhs, const DevTensor<T>& rhs) {
    auto R = lhs.copy();
    R += rhs;
    return R;
}

template <typename T>
inline DevTensor<T> operator+(DevTensor<T>&& lhs, const DevTensor<T>& rhs) {
    lhs += rhs;
    return std::move(lhs);
}

template <typename T>
inline DevTensor<T> operator+(const DevTensor<T>& lhs, DevTensor<T>&& rhs) {
    rhs += lhs;
    return std::move(rhs);
}

template <typename T>
inline DevTensor<T> operator+(DevTensor<T>&& lhs, DevTensor<T>&& rhs) {
    lhs += rhs;
    return std::move(lhs);
}

template <typename T>
inline DevTensor<T>& DevTensor<T>::operator-=(const DevTensor<T>& rhs) {
    assert(shape() == rhs.shape());
    gpgpu::blas::axpy(size(), T(-1), rhs.buffer(), 0, 1, buffer(), 0, 1, m_queue);
    return *this;
}

template <typename T>
inline DevTensor<T> operator-(const DevTensor<T>& lhs, const DevTensor<T>& rhs) {
    auto R = lhs.copy();
    R -= rhs;
    return R;
}

template <typename T>
inline DevTensor<T> operator-(DevTensor<T>&& lhs, const DevTensor<T>& rhs) {
    lhs -= rhs;
    return std::move(lhs);
}

template <typename T>
inline DevTensor<T> operator-(const DevTensor<T>& lhs, DevTensor<T>&& rhs) {
    auto R = lhs.copy(); // FIXME
    R -= rhs;
    return R;
}

template <typename T>
inline DevTensor<T> operator-(DevTensor<T>&& lhs, DevTensor<T>&& rhs) {
    lhs -= rhs;
    return lhs;
}

template <typename T>
inline DevTensor<T>& DevTensor<T>::operator*=(const DevTensor<T>& rhs) {
    gpgpu::blas::had(size(), T(1), buffer(), 0, 1, rhs.buffer(), 0, 1, T(0), buffer(), 0, 1, m_queue);
    return *this;
}

template <typename T>
inline DevTensor<T> operator*(const DevTensor<T>& lhs, const DevTensor<T>& rhs) {
    auto R = lhs.copy();
    R *= rhs;
    return R;
}

template <typename T>
inline DevTensor<T> operator*(DevTensor<T>&& lhs, const DevTensor<T>& rhs) {
    lhs *= rhs;
    return std::move(lhs);
}

template <typename T>
inline DevTensor<T> operator*(const DevTensor<T>& lhs, DevTensor<T>&& rhs) {
    rhs *= lhs;
    return std::move(rhs);
}

template <typename T>
inline DevTensor<T> operator*(DevTensor<T>&& lhs, DevTensor<T>&& rhs) {
    lhs *= rhs;
    return std::move(lhs);
}

template <typename T>
inline DevTensor<T>& DevTensor<T>::operator*=(const T& rhs) {
    gpgpu::blas::scal(size(), rhs, buffer(), 0, 1, m_queue);
    return *this;
}

template <typename T>
inline DevTensor<T> operator*(const DevTensor<T>& lhs, const T& rhs) {
    auto R = lhs.copy();
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
    auto R = rhs.copy();
    R *= lhs;
    return R;
}

template <typename T>
inline DevTensor<T> operator*(const T& lhs, DevTensor<T>&& rhs) {
    rhs *= lhs;
    return std::move(rhs);
}

/**
 * Perform inner product on two tensors. The tensors must be vector
 * or matrix and have compatible dimensions.
 */
template <typename T>
DevTensor<T>& inner(const DevTensor<T>& A, const DevTensor<T>& B, DevTensor<T>* C) {
    auto q = A.queue();
    assert(q == B.queue() && q == C->queue());

    if (A.is_vector() && B.is_vector()) {
        auto n = A.extent(0);
        assert(n == B.extent(0));
        assert(C->is_vector() && 1 == C->extent(0));
        gpgpu::blas::dot(n, A.buffer(), 0, 1, B.buffer(), 0, 1, C->buffer(), 0, q);
    } else if (A.is_matrix() && B.is_vector()) {
        auto m = A.extent(0), n = A.extent(1);
        assert(n == B.extent(0));
        assert(C->is_vector() && m == C->extent(0));
        gpgpu::blas::gemv(gpgpu::blas::Layout::RowMajor,
                          gpgpu::blas::Transpose::NoTrans,
                          m, n, T(1),
                          A.buffer(), 0, A.stride(0),
                          B.buffer(), 0, 1, T(0),
                          C->buffer(), 0, 1, q);
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

        gpgpu::blas::gemm(gpgpu::blas::Layout::RowMajor,
                          gpgpu::blas::Transpose::NoTrans,
                          gpgpu::blas::Transpose::NoTrans,
                          C_shape.extent(0), C_shape.extent(1), A_shape.extent(1),
                          T(1),
                          A.buffer(), 0, A_shape.stride(0),
                          B.buffer(), 0, B_shape.stride(0),
                          T(0),
                          C->buffer(), 0, C_shape.stride(0), q);
    } else {
        assert(false);
    }
    return *C;
}

template <typename T>
DevTensor<T> inner(const DevTensor<T>& A, const DevTensor<T>& B) {
    assert(A.queue() == B.queue());
    if (A.is_vector() && B.is_vector()) {
        assert(A.shape() == B.shape());
        DevTensor<T> C({1}, A.queue());
        inner(A, B, &C);
        return C;
    } else if (A.is_matrix() && B.is_vector()) {
        assert(A.extent(1) == B.extent(0));
        DevTensor<T> C({A.extent(0)}, A.queue());
        inner(A, B, &C);
        return C;
    } else if (A.is_vector() && B.is_matrix()) {
        assert(A.extent(0) == B.extent(0));
        DevTensor<T> C({B.extent(1)}, A.queue());
        inner(A, B, &C);
        return C;
    } else if (A.is_matrix() && B.is_matrix()) {
        auto m = A.extent(0), k = A.extent(1);
        auto p = B.extent(0), n = B.extent(1);
        assert(k == p);
        DevTensor<T> C({m, n}, A.queue());
        inner(A, B, &C);
        return C;
    } else {
        assert(false);
        return DevTensor<T>();
    }
}

/**
 * General matrix multiplication.
 */
template <typename T>
void gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
          const T& beta, DevTensor<T>* C,
          bool transA = false, bool transB = false)
{
    auto q = A.queue();
    assert(q == B.queue() && q == C->queue());

    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    assert(k == p);
    assert(C->shape() == Shape({m, n}));

    gpgpu::blas::gemm(gpgpu::blas::Layout::RowMajor,
                      transA ? gpgpu::blas::Transpose::Trans : gpgpu::blas::Transpose::NoTrans,
                      transB ? gpgpu::blas::Transpose::Trans : gpgpu::blas::Transpose::NoTrans,
                      m, n, k,
                      alpha,
                      A.buffer(), 0, A.stride(0),
                      B.buffer(), 0, B.stride(0),
                      beta,
                      C->buffer(), 0, C->stride(0), q);
}

template <typename T>
DevTensor<T> gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
                  const T& beta, DevTensor<T>& C,
                  bool transA = false, bool transB = false) {
    DevTensor<T> R = C.copy();
    gemm(alpha, A, B, beta, &R, transA, transB);
    return R;
}

} // namespace dlf
