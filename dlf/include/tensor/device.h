#pragma once

#include "tensor/shape.h"
#include "tensor/host.h"
#include "gpgpu.h"
#include "gblas.h"
#include "gdnn.h"

namespace dlf {

template <typename T> class DevTensorView;

/**
 * A tensor which data allocated from compute devices such as GPU.
 */
template <typename T>
class DevTensor : public Shaped {
    gpgpu::Buffer<T> m_data;

    friend class DevTensorView<T>;

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
        Shaped::resize(src.shape());
        src.copyTo(*this);
        return *this;
    }

    DevTensor(DevTensor&&) = default;
    DevTensor& operator=(DevTensor&&) = default;

    explicit DevTensor(const DevTensorView<T>& src);
    DevTensor& operator=(const DevTensorView<T>& src);

    DevTensor& resize(const Shape& shape) {
        if (empty()) {
            Shaped::resize(shape);
            m_data = gpgpu::current::context().createBuffer<T>(size());
        } else if (this->shape() != shape) {
            throw shape_error("incompatible shape");
        }
        return *this;
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, DevTensor&>
    resize(Args... args) {
        return resize({static_cast<size_t>(args)...});
    }

public:
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

public:
    const Shape& original_shape() const {
        return shape();
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

public: // Shape operations
    using Shaped::reshape;
    using Shaped::flatten;
    using Shaped::squeeze;
    using Shaped::unsqueeze;

    DevTensorView<T> broadcast(const Shape& to) const;
    DevTensorView<T> transpose(const std::vector<size_t>& perm) const;
    DevTensorView<T> transpose() const;
    DevTensorView<T> slice(
        const std::vector<int>& starts, const std::vector<int>& ends,
        const std::vector<int>& axes, const std::vector<int>& steps) const;
    DevTensorView<T> slice(const std::vector<SliceDim>& dims) const;
    DevTensorView<T> diagonal() const;

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, DevTensorView<T>>
    transpose(Args... args) const {
        return transpose({static_cast<size_t>(args)...});
    }

    /**
     * We use ~ operator to represent tensor transposition instead of bitwise-not
     * operator.
     */
    DevTensorView<T> operator~() const {
        return transpose();
    }

    DevTensorView<T> operator[](const std::vector<SliceDim>& dims) const {
        return slice(dims);
    }
};

template <typename T>
inline DevTensor<T> dev(const Tensor<T>& host) {
    return DevTensor<T>(host);
}

template <typename T>
class DevTensorView : public Shaped {
    Shape m_original_shape;
    gpgpu::Buffer<T> m_data;

public:
    DevTensorView(Shape shape, const DevTensor<T>& src);
    DevTensorView(Shape shape, const DevTensorView<T>& src);

public:
    const Shape& original_shape() const noexcept {
        return m_original_shape;
    }

    gpgpu::Buffer<T>& data() noexcept {
        return m_data;
    }

    const gpgpu::Buffer<T>& data() const noexcept {
        return m_data;
    }

    operator DevTensor<T>() const {
        return DevTensor<T>(*this);
    }

    DevTensor<T> reorder() const {
        return DevTensor<T>(*this);
    }

    Tensor<T> read() const {
        return DevTensor<T>(*this).read();
    }

public: // Shape operations
    DevTensorView<T> broadcast(const Shape& to) const;
    DevTensorView<T> transpose(const std::vector<size_t>& perm) const;
    DevTensorView<T> transpose() const;
    DevTensorView<T> slice(
        const std::vector<int>& starts, const std::vector<int>& ends,
        const std::vector<int>& axes, const std::vector<int>& steps) const;
    DevTensorView<T> slice(const std::vector<SliceDim>& dims) const;
    DevTensorView<T> diagonal() const;

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, DevTensorView<T>>
    transpose(Args... args) const {
        return transpose({static_cast<size_t>(args)...});
    }

    DevTensorView<T> operator~() const {
        return transpose();
    }

    DevTensorView<T> operator[](const std::vector<SliceDim>& dims) const {
        return slice(dims);
    }
};

template <typename T>
DevTensorView<T>::DevTensorView(Shape shape, const DevTensor<T>& src)
    : Shaped(std::move(shape), true),
      m_original_shape(src.original_shape()),
      m_data(src.m_data)
{}

template <typename T>
DevTensorView<T>::DevTensorView(Shape shape, const DevTensorView<T>& src)
    : Shaped(std::move(shape), true),
      m_original_shape(src.original_shape()),
      m_data(src.m_data)
{}

template <typename T>
DevTensor<T>::DevTensor(const DevTensorView<T>& src) {
    reorder(src, *this);
}

template <typename T>
DevTensor<T>& DevTensor<T>::operator=(const DevTensorView<T>& src) {
    if (size() != src.size())
        m_data = gpgpu::current::context().createBuffer<T>(src.size());
    Shaped::resize(src.shape());
    reorder(src, *this);
    return *this;
}

template <typename T>
inline void flat_copy(const DevTensor<T>& src, DevTensor<T>& dst) {
    assert(src.size() == dst.size());
    src.copyToAsync(dst);
}

//==-------------------------------------------------------------------------
// DevTensor shape operations
//==-------------------------------------------------------------------------

template <typename T>
inline DevTensorView<T> DevTensor<T>::broadcast(const Shape& to) const {
    return DevTensorView<T>(shape().broadcast(to), *this);
}

template <typename T>
inline DevTensorView<T> DevTensorView<T>::broadcast(const Shape& to) const {
    return DevTensorView<T>(shape().broadcast(to), *this);
}

template <typename T>
inline DevTensorView<T> DevTensor<T>::transpose(const std::vector<size_t>& perm) const {
    return DevTensorView<T>(shape().transpose(perm), *this);
}

template <typename T>
DevTensorView<T> DevTensor<T>::transpose() const {
    return DevTensorView<T>(shape().transpose(), *this);
}

template <typename T>
inline DevTensorView<T> DevTensorView<T>::transpose(const std::vector<size_t>& perm) const {
    return DevTensorView<T>(shape().transpose(perm), *this);
}

template <typename T>
DevTensorView<T> DevTensorView<T>::transpose() const {
    return DevTensorView<T>(shape().transpose(), *this);
}

template <typename T>
inline DevTensorView<T> DevTensor<T>::slice(
    const std::vector<int>& starts, const std::vector<int>& ends,
    const std::vector<int>& axes, const std::vector<int>& steps) const
{
    return DevTensorView<T>(shape().slice(starts, ends, axes, steps), *this);
}

template <typename T>
DevTensorView<T> DevTensor<T>::slice(const std::vector<SliceDim>& dims) const {
    return DevTensorView<T>(shape().slice(dims), *this);
}

template <typename T>
inline DevTensorView<T> DevTensorView<T>::slice(
    const std::vector<int>& starts, const std::vector<int>& ends,
    const std::vector<int>& axes, const std::vector<int>& steps) const
{
    return DevTensorView<T>(shape().slice(starts, ends, axes, steps), *this);
}

template <typename T>
DevTensorView<T> DevTensorView<T>::slice(const std::vector<SliceDim>& dims) const {
    return DevTensorView<T>(shape().slice(dims), *this);
}

template <typename T>
DevTensorView<T> DevTensor<T>::diagonal() const {
    return DevTensorView<T>(shape().diagonal(), *this);
}

template <typename T>
DevTensorView<T> DevTensorView<T>::diagonal() const {
    return DevTensorView<T>(shape().diagonal(), *this);
}

template <typename T>
inline DevTensorView<T> squeeze(const DevTensorView<T>& src, const std::vector<int>& axes = {}) {
    return DevTensorView<T>(src.shape().squeeze(axes), src);
}

template <typename T>
inline DevTensorView<T> unsqueeze(const DevTensorView<T>& src, const std::vector<int>& axes) {
    return DevTensorView<T>(src.shape().unsqueeze(axes), src);
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
        C->resize({1});
        gblas::dot(n, A.data(), 1, B.data(), 1, C->data());
        return *C;
    }

    if (A.is_matrix() && B.is_vector()) {
        auto m = A.extent(0), n = A.extent(1);
        assert(n == B.extent(0));
        C->resize({m});
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
            A_shape = Shape({1, A.extent(0)});
            B_shape = B.shape();
            C_shape = Shape({1, B.extent(1)});
            C->resize({B.extent(1)});
        } else {
            A_shape = A.shape();
            B_shape = B.shape();
            C_shape = Shape({A.extent(0), B.extent(1)});
            C->resize(C_shape);
        }

        assert(A_shape.extent(1) == B_shape.extent(0));
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
    if (k != p)
        throw shape_error("gemm: incompatible shape");
    C->resize({m, n});

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
