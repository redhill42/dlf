#ifndef _TENSOR_DEVICE_H
#define _TENSOR_DEVICE_H

#include "tensor/shape.h"
#include "tensor/host.h"
#include "gpgpu.h"
#include "gpblas.h"

namespace tensor {

/**
 * A tensor which data allocated from compute devices such as GPU.
 */
template <typename T>
class DevTensor {
    Shape m_shape;
    size_t m_size = 0;

    mutable gpgpu::Queue m_queue;
    mutable gpgpu::Buffer<T> m_data;

public:
    DevTensor() = default;

    DevTensor(Shape shape, gpgpu::Queue queue)
        : m_shape(std::move(shape)), m_queue(std::move(queue))
    {
        m_size = m_shape.size();
        m_data = m_queue.context().template createBuffer<T>(m_size);
    }

    DevTensor(const Tensor<T>& host, gpgpu::Queue queue)
        : m_shape(host.shape()), m_queue(std::move(queue))
    {
        m_size = m_shape.size();
        m_data = m_queue.context().template createBuffer<T>(m_size);
        m_data.write(m_queue, host.data(), host.size());
    }

    DevTensor(Shape shape, gpgpu::Buffer<T> data, gpgpu::Queue queue)
        : m_shape(std::move(shape)), m_data(std::move(data)), m_queue(std::move(queue))
    {
        m_size = m_shape.size();
    }

    const Shape& shape() const noexcept {
        return m_shape;
    }

    bool reshape(Shape newshape) {
        return m_shape.reshape(std::move(newshape));
    }

    /**
     * Returns the total size of this tensor.
     */
    size_t size() const noexcept {
        return m_size;
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
     * Returns true if this tensor is a square matrix.
     */
    bool is_square() const noexcept {
        return is_matrix() && m_shape[0] == m_shape[1];
    }

    /**
     * Read data from device.
     */
    Tensor<T> read() const {
        Tensor<T> host(m_shape);
        m_data.read(m_queue, host.data(), host.size());
        return host;
    }

    /**
     * Asynchronously read data from deivce.
     */
    Tensor<T> readAsync() const {
        Tensor<T> host(m_shape);
        m_data.readAsync(m_queue, host.data(), host.size());
        return host;
    };

    /**
     * Read data from device and store data into given host tensor.
     */
    void read(Tensor<T>& host) const {
        assert(m_shape == host.shape());
        m_data.read(m_queue, host.data(), host.size());
    }

    /**
     * Asynchronously read data from device and store data into given host tensor.
     */
    void readAsync(Tensor<T>& host) const {
        assert(m_shape == host.shape());
        m_data.readAsync(m_queue, host.data(), host.size());
    }

    /**
     * Write data into device.
     */
    void write(const Tensor<T>& host) {
        assert(m_shape == host.shape());
        m_data.write(m_queue, host.data(), host.size());
    }

    /**
     * Asynchronously write data into device.
     */
    void writeAsync(const Tensor<T>& host) {
        assert(m_shape == host.shape());
        m_data.writeAsync(m_queue, host.data(), host.size());
    }

    /**
     * Copy data into target.
     */
    void copyTo(DevTensor<T>& dest) const {
        assert(m_shape == dest.shape());
        m_data.copyTo(m_queue, dest.buffer(), size());
    }

    /**
     * Asynchronously copy data into destination.
     */
    void copyToAsync(DevTensor<T>& dest) const {
        assert(m_shape == dest.shape());
        m_data.copyToAsync(m_queue, dest.buffer(), size());
    }

    /**
     * Create a copy of this tensor.
     */
    DevTensor<T> copy() const {
        DevTensor<T> dest(m_shape, m_queue);
        m_data.copyTo(m_queue, dest.buffer(), size());
        return dest;
    }

    /**
     * Asynchronously create a copy of this tensor.
     */
    DevTensor<T> copyAsync() const {
        DevTensor<T> dest(m_shape, m_queue);
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
    assert(m_shape == rhs.shape());
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
    assert(m_shape == rhs.shape());
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
void inner(const DevTensor<T>& A, const DevTensor<T>& B, DevTensor<T>* C) {
    auto q = A.queue();
    assert(q == B.queue() && q == C->queue());

    if (B.is_vector()) {
        auto n = B.shape()[0];

        // vector . vector
        if (A.is_vector()) {
            assert(n == A.shape()[0]);
            assert(C->is_vector() && 1 == C->shape()[0]);
            gpgpu::blas::dot(n, A.buffer(), 0, 1, B.buffer(), 0, 1, C->buffer(), 0, q);
            return;
        }

        // matrix . vector
        if (A.is_matrix()) {
            auto m = A.shape()[0];
            assert(n == A.shape()[1]);
            assert(C->is_vector() && m == C->shape()[0]);
            gpgpu::blas::gemv(gpgpu::blas::Layout::RowMajor,
                              gpgpu::blas::Transpose::NoTrans,
                              m, n, T(1),
                              A.buffer(), 0, n,
                              B.buffer(), 0, 1, T(0),
                              C->buffer(), 0, 1, q);
            return;
        }

        assert(false);
    }

    if (B.is_matrix()) {
        auto [k, n] = B.shape().extent();
        size_t m = 0;

        if (A.is_vector()) {
            // vector . matrix --> (1xk matrix) . matrix
            m = 1;
            assert(k == A.shape()[0]);
            assert(C->is_vector() && n == C->shape()[0]);
        } else if (A.is_matrix()) {
            // matrix . matrix
            m = A.shape()[0];
            assert(k == A.shape()[1]);
            assert(C->shape() == Shape({m, n}));
        } else {
            assert(false);
            return;
        }

        gpgpu::blas::gemm(gpgpu::blas::Layout::RowMajor,
                          gpgpu::blas::Transpose::NoTrans,
                          gpgpu::blas::Transpose::NoTrans,
                          m, n, k,
                          T(1), A.buffer(), 0, k, B.buffer(), 0, n,
                          T(0), C->buffer(), 0, n, q);
    }
}

template <typename T>
DevTensor<T> inner(const DevTensor<T>& A, const DevTensor<T>& B) {
    assert(A.queue() == B.queue());

    if (B.is_vector()) {
        if (A.is_vector()) {
            assert(A.shape() == B.shape());
            DevTensor<T> C({1}, A.queue());
            inner(A, B, &C);
            return C;
        }

        if (A.is_matrix()) {
            assert(A.shape()[1] == B.shape()[0]);
            DevTensor<T> C({A.shape()[0]}, A.queue());
            inner(A, B, &C);
            return C;
        }

        assert(false);
        return DevTensor<T>();
    }

    if (B.is_matrix()) {
        if (A.is_vector()) {
            assert(A.shape()[0] == B.shape()[0]);
            DevTensor<T> C({B.shape()[1]}, A.queue());
            inner(A, B, &C);
            return C;
        }

        if (A.is_matrix()) {
            assert(A.shape()[1] == B.shape()[0]);
            DevTensor<T> C({A.shape()[0], B.shape()[1]}, A.queue());
            inner(A, B, &C);
            return C;
        }
    }

    assert(false);
    return DevTensor<T>();
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
    auto [m, k] = A.shape().extent();
    if (transA)
        std::swap(m, k);

    auto [p, n] = B.shape().extent();
    if (transB)
        std::swap(p, n);

    assert(k == p);
    assert(C->shape() == Shape({m, n}));

    gpgpu::blas::gemm(gpgpu::blas::Layout::RowMajor,
                      transA ? gpgpu::blas::Transpose::Trans : gpgpu::blas::Transpose::NoTrans,
                      transB ? gpgpu::blas::Transpose::Trans : gpgpu::blas::Transpose::NoTrans,
                      m, n, k,
                      alpha,
                      A.buffer(), 0, A.shape()[1],
                      B.buffer(), 0, B.shape()[1],
                      beta,
                      C->buffer(), 0, n, q);
}

template <typename T>
DevTensor<T> gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
                  const T& beta, DevTensor<T>& C,
                  bool transA = false, bool transB = false) {
    DevTensor<T> R = C.copy();
    gemm(alpha, A, B, beta, &R, transA, transB);
    return R;
}

} // namespace tensor

#endif //_TENSOR_DEVICE_H
