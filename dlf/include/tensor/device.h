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

    DevTensor& resize(const Shape& shape);

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

public:
    /**
     * Create a scalar.
     */
    static DevTensor scalar(const T& value);

    /**
     * Create an identity tensor.
     */

    static DevTensor identity(Shape shape, const T& value = T{1});

    /**
     * Fill data fill scalar value.
     */
    DevTensor& fill(const T& value) &;
    DevTensor fill(const T& value) &&;

    /**
     * Fill data containing a sequence of numbers that begin at start
     * and extends by increments of delta.
     */
    DevTensor& range(T start = 0, T delta = 1) &;
    DevTensor range(T start = 0, T delta = 1) &&;

    /**
     * Fill then tensor with random data.
     */
    template <typename D>
    std::enable_if_t<is_random_distribution_type<T, D>::value, DevTensor&>
    random(D&& d) &;

    template <typename D>
    std::enable_if_t<is_random_distribution_type<T, D>::value, DevTensor>
    random(D&& d) &&;

    /**
     * Fill the tensor with random data with uniform distribution.
     */
    DevTensor& random(T low = 0, T high = std::numeric_limits<T>::max()) &;
    DevTensor random(T low = 0, T high = std::numeric_limits<T>::max()) &&;

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

    DevTensorView& resize(const Shape& shape) {
        if (this->shape() != shape)
            throw shape_error("incompatible shape");
        return *this;
    }

    template <typename... Args>
    std::enable_if_t<cxx::conjunction<std::is_integral<Args>...>::value, DevTensorView&>
    resize(Args... args) {
        return resize({static_cast<size_t>(args)...});
    }

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

public:
    DevTensorView& fill(const T& value);
    DevTensorView& range(T start = 0, T delta = 1);

    template <typename D>
    std::enable_if_t<is_random_distribution_type<T, D>::value, DevTensorView&>
    random(D&& d);

    DevTensorView& random(T low = 0, T high = std::numeric_limits<T>::max());
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
DevTensor<T>& DevTensor<T>::resize(const Shape& shape) {
    if (this->empty() || this->shape() != shape) {
        Spatial<DevTensor>::set_shape(shape);
        if (m_data.handle() == nullptr || m_data.size() < size())
            init();
    }
    return *this;
}

template <typename T>
inline void flat_copy(const DevTensor<T>& src, DevTensor<T>& dst) {
    assert(src.size() == dst.size());
    src.copyToAsync(dst);
}

template <typename T>
inline void flat_copy(const DevTensor<T>& src, DevTensor<T>&& dst) {
    assert(src.size() == dst.size());
    src.copyToAsync(dst);
}

template <typename T>
DevTensor<T> DevTensor<T>::scalar(const T& value) {
    DevTensor<T> res{Shape()};
    res.data().write(gpgpu::current::queue(), &value, 1);
    return res;
}

template <typename T>
DevTensor<T> DevTensor<T>::identity(Shape shape, const T& value) {
    DevTensor res(std::move(shape), T{});
    res.diagonal().fill(value);
    return res;
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
inline DevTensor<T>& DevTensor<T>::range(T start, T delta) & {
    gpgpu::dnn::range(size(), start, delta, data(), 0);
    return *this;
}

template <typename T>
inline DevTensor<T> DevTensor<T>::range(T start, T delta) && {
    gpgpu::dnn::range(size(), start, delta, data(), 0);
    return std::move(*this);
}

template <typename T>
DevTensorView<T>& DevTensorView<T>::range(T start, T delta) {
    if (shape().is_contiguous()) {
        gpgpu::dnn::range(size(), start, delta, data(), shape().offset());
    } else {
        gpgpu::dnn::range(size(), start, delta, shape().extents(), shape().strides(), data(), shape().offset());
    }
    return *this;
}

namespace detail {
template <typename D>
struct is_uniform_distribution : std::false_type {};

template <typename T>
struct is_uniform_distribution<std::uniform_int_distribution<T>> : std::true_type {};

template <typename T>
struct is_uniform_distribution<std::uniform_real_distribution<T>> : std::true_type {};

template <typename T, typename TensorT, typename D>
std::enable_if_t<is_uniform_distribution<D>::value, TensorT&>
gpu_randomize(TensorT& t, D&& d) {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> rng;
    gpgpu::dnn::random(
        t.size(), t.shape().extents(), t.shape().strides(),
        t.data(), t.shape().offset(), rng(rd),
        static_cast<T>(d.a()), static_cast<T>(d.b()));
    return t;
}

template <typename T, typename TensorT, typename U>
TensorT& gpu_randomize(TensorT& t, const std::normal_distribution<U>& d) {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> rng;
    gpgpu::dnn::random_normal(
        t.size(), t.shape().extents(), t.shape().strides(),
        t.data(), t.shape().offset(), rng(rd),
        static_cast<T>(d.mean()), static_cast<T>(d.stddev()));
    return t;
}

template <typename TensorT, typename T>
TensorT& gpu_randomize(TensorT& t, T low, T high) {
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> rng;
    gpgpu::dnn::random(
        t.size(), t.shape().extents(), t.shape().strides(),
        t.data(), t.shape().offset(), rng(rd), low, high);
    return t;
}
} // namespace detail

template <typename T>
template <typename D>
std::enable_if_t<is_random_distribution_type<T, D>::value, DevTensor<T>&>
inline DevTensor<T>::random(D&& d) & {
    return detail::gpu_randomize<T>(*this, std::forward<D>(d));
}

template <typename T>
template <typename D>
std::enable_if_t<is_random_distribution_type<T, D>::value, DevTensor<T>>
inline DevTensor<T>::random(D&& d) && {
    return std::move(detail::gpu_randomize<T>(*this, std::forward<D>(d)));
}

template <typename T>
inline DevTensor<T>& DevTensor<T>::random(T low, T high) & {
    return detail::gpu_randomize(*this, low, high);
}

template <typename T>
inline DevTensor<T> DevTensor<T>::random(T low, T high) && {
    return std::move(detail::gpu_randomize(*this, low, high));
}

template <typename T>
template <typename D>
std::enable_if_t<is_random_distribution_type<T, D>::value, DevTensorView<T>&>
inline DevTensorView<T>::random(D&& d) {
    return detail::gpu_randomize<T>(*this, std::forward<D>(d));
}

template <typename T>
inline DevTensorView<T>& DevTensorView<T>::random(T low, T high) {
    return detail::gpu_randomize(*this, low, high);
}

template <typename T, typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, const DevTensor<T>& t) {
    return out << t.read();
}

template <typename T, typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, const DevTensorView<T>& v) {
    return out << v.read();
}

} // namespace dlf
