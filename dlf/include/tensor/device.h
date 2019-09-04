#pragma once

#include "gpgpu.h"
#include "gblas.h"
#include "gdnn.h"

namespace dlf {

template <typename T> class DevTensorView;

/**
 * A tensor which data allocated from compute devices such as GPU.
 */
template <typename T>
class DevTensor : public Spatial<DevTensor<T>> {
    gpgpu::Buffer<T> m_data;

    void init() {
        assert(size() != 0);
        m_data = gpgpu::current::context().createBuffer<T>(size());
    }

    friend class DevTensorView<T>;

public:
    DevTensor() = default;

    explicit DevTensor(Shape shape) : Spatial<DevTensor>(std::move(shape)) {
        init();
    }

    explicit DevTensor(Shape shape, const T& initial) : Spatial<DevTensor>(std::move(shape)) {
        init();
        fill(initial);
    }

    explicit DevTensor(const Tensor<T>& host) : Spatial<DevTensor>(host.shape()) {
        init();
        m_data.write(gpgpu::current::queue(), host.data(), host.size());
    }

    explicit DevTensor(Shape shape, gpgpu::Buffer<T> data)
        : Spatial<DevTensor>(std::move(shape)), m_data(std::move(data))
    {}

    DevTensor(const DevTensor& src) : Spatial<DevTensor>(src) {
        init();
        src.copyTo(*this);
    }

    DevTensor& operator=(const DevTensor& src) {
        auto old_size = size();
        Spatial<DevTensor>::set_shape(src.shape());
        if (size() != old_size || m_data.handle() == nullptr)
            init();
        src.copyTo(*this);
        return *this;
    }

    DevTensor(DevTensor&&) = default;
    DevTensor& operator=(DevTensor&&) = default;

    explicit DevTensor(const DevTensorView<T>& src);
    DevTensor& operator=(const DevTensorView<T>& src);

    DevTensor& resize(const Shape& shape) {
        if (this->empty()) {
            Spatial<DevTensor>::set_shape(shape);
            init();
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
    using Spatial<DevTensor>::shape;
    using Spatial<DevTensor>::size;

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
     * Returns a view of this tensor with given shape.
     */
    DevTensorView<T> view(Shape shape) const {
        return DevTensorView<T>(std::move(shape), *this);
    }

    /**
     * Create a scalar.
     */
    static DevTensor scalar(const T& value) {
        DevTensor<T> res{Shape()};
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
    using Spatial<DevTensor>::reshape;
    using Spatial<DevTensor>::flatten;
    using Spatial<DevTensor>::squeeze;
    using Spatial<DevTensor>::unsqueeze;
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
class DevTensorView : public Spatial<DevTensorView<T>> {
    Shape m_original_shape;
    gpgpu::Buffer<T> m_data;

public:
    DevTensorView() = default;
    DevTensorView(Shape shape, const DevTensor<T>& src);
    DevTensorView(Shape shape, const DevTensorView<T>& src);

public:
    using Spatial<DevTensorView>::shape;
    using Spatial<DevTensorView>::size;

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

    DevTensorView view(Shape shape) const {
        return DevTensorView(std::move(shape), *this);
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
};

template <typename T>
DevTensorView<T>::DevTensorView(Shape shape, const DevTensor<T>& src)
    : Spatial<DevTensorView>(std::move(shape), true),
      m_original_shape(src.original_shape()),
      m_data(src.m_data)
{}

template <typename T>
DevTensorView<T>::DevTensorView(Shape shape, const DevTensorView<T>& src)
    : Spatial<DevTensorView>(std::move(shape), true),
      m_original_shape(src.original_shape()),
      m_data(src.m_data)
{}

template <typename T>
DevTensor<T>::DevTensor(const DevTensorView<T>& src) {
    reorder(src, *this);
}

template <typename T>
DevTensor<T>& DevTensor<T>::operator=(const DevTensorView<T>& src) {
    auto old_size = size();
    Spatial<DevTensor>::set_shape(src.shape());
    if (size() != old_size || m_data.handle() == nullptr)
        init();
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
