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

    explicit DevTensor(Shape shape) : Shaped(std::move(shape)) {
        m_data = gpgpu::current::context().createBuffer<T>(size());
    }

    explicit DevTensor(Shape shape, const T& initial) : Shaped(std::move(shape)) {
        m_data = gpgpu::current::context().createBuffer<T>(size());
        fill(initial);
    }

    explicit DevTensor(const Tensor<T>& host) : Shaped(host.shape()) {
        m_data = gpgpu::current::context().createBuffer<T>(size());
        m_data.write(gpgpu::current::queue(), host.data(), host.size());
    }

    explicit DevTensor(Shape shape, gpgpu::Buffer<T> data)
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
     * Returns a view of this tensor.
     */
    DevTensorView<T> view() const {
        return DevTensorView<T>(shape(), *this);
    }

    /**
     * Create a scalar.
     */
    static DevTensor scalar(const T& value) {
        DevTensor<T> res({1});
        res.data().write(gpgpu::current::queue(), &value, 1);
        return res;
    }

public:
    /**
     * Fill data fill scalar value.
     */
    DevTensor& fill(const T& value) &;
    DevTensor fill(const T& value) &&;

    /**
     * Create an identity tensor.
     */
    static DevTensor identity(Shape shape, const T& value = T{1});

public: // Shape operations
    using Shaped::reshape;
    using Shaped::flatten;
    using Shaped::squeeze;
    using Shaped::unsqueeze;

    DevTensorView<T> broadcast(const Shape& to) const {
        return DevTensorView<T>(shape().broadcast(to), *this);
    }

    DevTensorView<T> transpose(const std::vector<size_t>& perm) const {
        return DevTensorView<T>(shape().transpose(perm), *this);
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, DevTensorView<T>>
    transpose(Args... args) const {
        return transpose({static_cast<size_t>(args)...});
    }

    DevTensorView<T> transpose() const {
        return DevTensorView<T>(shape().transpose(), *this);
    }

    DevTensorView<T> operator~() const {
        return transpose();
    }

    DevTensorView<T> slice(
        const std::vector<int>& starts, const std::vector<int>& ends,
        const std::vector<int>& axes, const std::vector<int>& steps) const
    {
        return DevTensorView<T>(shape().slice(starts, ends, axes, steps), *this);
    }

    DevTensorView<T> slice(const std::vector<SliceDim>& dims) const {
        return DevTensorView<T>(shape().slice(dims), *this);
    }

    DevTensorView<T> operator[](const char* spec) const {
        return DevTensorView<T>(shape().slice(spec), *this);
    }

    DevTensorView<T> operator[](const std::string& spec) const {
        return operator[](spec.c_str());
    }

    DevTensorView<T> operator[](const int index) const {
        return TensorView<T>(shape().slice({{index, index+1}}).squeeze(0), *this);
    }

    DevTensorView<T> diagonal(int offset = 0, int axis1 = -2, int axis2 = -1) const {
        return DevTensorView<T>(shape().diagonal(offset, axis1, axis2), *this);
    }
};

template <typename T>
inline DevTensor<T> dev(const Tensor<T>& host) {
    return DevTensor<T>(host);
}

template <typename T>
inline DevTensor<T> dev(const TensorView<T>& host) {
    return DevTensor<T>(host.reorder());
}

template <typename T>
class DevTensorView : public Shaped {
    Shape m_original_shape;
    gpgpu::Buffer<T> m_data;

public:
    DevTensorView() = default;
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

    DevTensorView view() const {
        return *this;
    }

    DevTensor<T> copy() const {
        return DevTensor<T>(*this);
    }

    DevTensor<T> reorder() const {
        if (shape().is_contiguous() && shape().offset() == 0) {
            return DevTensor<T>(shape(), m_data);
        } else {
            return copy();
        }
    }

    operator DevTensor<T>() const {
        return reorder();
    }

    Tensor<T> read() const {
        return reorder().read();
    }

    DevTensorView& fill(const T& value);

public: // Shape operations
    DevTensorView<T> broadcast(const Shape& to) const {
        return DevTensorView<T>(shape().broadcast(to), *this);
    }

    DevTensorView<T> transpose(const std::vector<size_t>& perm) const {
        return DevTensorView<T>(shape().transpose(perm), *this);
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, DevTensorView<T>>
    transpose(Args... args) const {
        return transpose({static_cast<size_t>(args)...});
    }

    DevTensorView<T> transpose() const {
        return DevTensorView<T>(shape().transpose(), *this);
    }

    DevTensorView<T> operator~() const {
        return transpose();
    }

    DevTensorView<T> slice(
        const std::vector<int>& starts, const std::vector<int>& ends,
        const std::vector<int>& axes, const std::vector<int>& steps) const
    {
        return DevTensorView<T>(shape().slice(starts, ends, axes, steps), *this);
    }

    DevTensorView<T> slice(const std::vector<SliceDim>& dims) const {
        return DevTensorView<T>(shape().slice(dims), *this);
    }

    DevTensorView<T> operator[](const char* spec) const {
        return DevTensorView<T>(shape().slice(spec), *this);
    }

    DevTensorView<T> operator[](const std::string& spec) const {
        return operator[](spec.c_str());
    }

    DevTensorView<T> operator[](const int index) const {
        return TensorView<T>(shape().slice({{index, index+1}}).squeeze(0), *this);
    }

    DevTensorView<T> diagonal(int offset = 0, int axis1 = -2, int axis2 = -1) const {
        return DevTensorView<T>(shape().diagonal(offset, axis1, axis2), *this);
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

template <typename T>
inline DevTensor<T>& DevTensor<T>::fill(const T& value) & {
    gpgpu::dnn::fill(size(), data(), 0, value);
    return *this;
}

template <typename T>
inline DevTensor<T> DevTensor<T>::fill(const T& value) && {
    gpgpu::dnn::fill(size(), data(), 0, value);
    return std::move(*this);
}

template <typename T>
DevTensorView<T>& DevTensorView<T>::fill(const T& value) {
    if (shape().is_contiguous()) {
        gpgpu::dnn::fill(size(), data(), shape().offset(), value);
    } else {
        gpgpu::dnn::fill(size(), shape().extents(), shape().strides(), data(), shape().offset(), value);
    }
    return *this;
}

template <typename T>
DevTensor<T> DevTensor<T>::identity(Shape shape, const T& value) {
    DevTensor res(std::move(shape), T{});
    res.diagonal().fill(value);
    return res;
}

} // namespace dlf
