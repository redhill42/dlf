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
    gpgpu::Buffer<T> m_data;

public:
    DevTensorView(Shape shape, const DevTensor<T>& src);
    DevTensorView(Shape shape, const DevTensorView<T>& src);

public:
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
      m_data(src.m_data)
{}

template <typename T>
DevTensorView<T>::DevTensorView(Shape shape, const DevTensorView<T>& src)
    : Shaped(std::move(shape), true),
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

//==-------------------------------------------------------------------------
// DevTensor unary transformations
//==-------------------------------------------------------------------------

namespace detail {
template <typename TensorX, typename TensorY>
void transform(const std::string& name, const TensorX& X, TensorY& Y) {
    assert(X.shape() == Y.shape());
    if (X.shape().is_contiguous() && Y.shape().is_contiguous()) {
        gpgpu::dnn::transform(name, X.size(),
                              X.data(), X.shape().offset(),
                              Y.data(), Y.shape().offset());
    } else {
        gpgpu::dnn::transform(name, X.size(), X.shape().extents(),
                              X.data(), X.shape().offset(), X.shape().strides(),
                              Y.data(), Y.shape().offset(), Y.shape().strides());
    }
}

template <typename T, typename TensorX, typename TensorY>
void transform(const std::string name, const T alpha, const T beta,
               const TensorX& X, TensorY& Y)
{
    assert(X.shape() == Y.shape());
    if (X.shape().is_contiguous() && Y.shape().is_contiguous()) {
        gpgpu::dnn::transform(name, alpha, beta, X.size(),
                              X.data(), X.shape().offset(),
                              Y.data(), Y.shape().offset());
    } else {
        gpgpu::dnn::transform(name, alpha, beta, X.size(), X.shape().extents(),
                              X.data(), X.shape().offset(), X.shape().strides(),
                              Y.data(), Y.shape().offset(), Y.shape().strides());
    }
}
} // namespace detail

template <typename T, typename Fn>
std::enable_if_t<!std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensor<T>&>
inline transformTo(const DevTensor<T>& X, DevTensor<T>& Y, Fn) {
    Y.resize(X.shape());
    detail::transform(Fn::name, X, Y);
    return Y;
}

template <typename T, typename Fn>
std::enable_if_t<!std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensorView<T>&>
inline transformTo(const DevTensor<T>& X, DevTensorView<T>& Y, Fn) {
    assert(Y.shape() == X.shape());
    detail::transform(Fn::name, X, Y);
    return Y;
}

template <typename T, typename Fn>
std::enable_if_t<!std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensorView<T>&>
inline transformTo(const DevTensor<T>& X, DevTensorView<T>&& Y, Fn fn) {
    return transformTo(X, Y, fn);
}

template <typename T, typename Fn>
std::enable_if_t<!std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensor<T>&>
inline transformTo(const DevTensorView<T>& X, DevTensor<T>& Y, Fn) {
    Y.resize(X.shape());
    detail::transform(Fn::name, X, Y);
    return Y;
}

template <typename T, typename Fn>
std::enable_if_t<!std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensorView<T>&>
inline transformTo(const DevTensorView<T>& X, DevTensorView<T>& Y, Fn) {
    assert(Y.shape() == X.shape());
    detail::transform(Fn::name, X, Y);
    return Y;
}

template <typename T, typename Fn>
std::enable_if_t<!std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensorView<T>&>
inline transformTo(const DevTensorView<T>& X, DevTensorView<T>&& Y, Fn fn) {
    return transformTo(X, Y, fn);
}

template <typename T, typename Fn>
std::enable_if_t<std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensor<T>&>
inline transformTo(const DevTensor<T>& X, DevTensor<T>& Y, Fn fn) {
    Y.resize(X.shape());
    detail::transform(Fn::name, fn.alpha, fn.beta, X, Y);
    return Y;
}

template <typename T, typename Fn>
std::enable_if_t<std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensorView<T>&>
inline transformTo(const DevTensor<T>& X, DevTensorView<T>& Y, Fn fn) {
    assert(Y.shape() == X.shape());
    detail::transform(Fn::name, fn.alpha, fn.beta, X, Y);
    return Y;
}

template <typename T, typename Fn>
std::enable_if_t<std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensorView<T>&>
inline transformTo(const DevTensor<T>& X, DevTensorView<T>&& Y, Fn fn) {
    return transformTo(X, Y, fn);
}

template <typename T, typename Fn>
std::enable_if_t<std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensor<T>&>
inline transformTo(const DevTensorView<T>& X, DevTensor<T>& Y, Fn fn) {
    Y.resize(X.shape());
    detail::transform(Fn::name, fn.alpha, fn.beta, X, Y);
    return Y;
}

template <typename T, typename Fn>
std::enable_if_t<std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensorView<T>&>
inline transformTo(const DevTensorView<T>& X, DevTensorView<T>& Y, Fn fn) {
    assert(Y.shape() == X.shape());
    detail::transform(Fn::name, fn.alpha, fn.beta, X, Y);
    return Y;
}

template <typename T, typename Fn>
std::enable_if_t<std::is_base_of<xfn::parameterized_function<T>, Fn>::value, DevTensorView<T>&>
inline transformTo(const DevTensorView<T>& X, DevTensorView<T>&& Y, Fn fn) {
    return transformTo(X, Y, fn);
}

template <typename T, typename Fn>
inline DevTensor<T> transform(const DevTensor<T>& X, Fn fn) {
    DevTensor<T> Y;
    transformTo(X, Y, fn);
    return Y;
}

template <typename T, typename Fn>
inline DevTensor<T> transform(DevTensor<T>&& X, Fn fn) {
    return std::move(transformTo(X, X, fn));
}

template <typename T, typename Fn>
inline DevTensor<T> transform(const DevTensorView<T>& X, Fn fn) {
    DevTensor<T> Y;
    transformTo(X, Y, fn);
    return Y;
}

namespace detail {
template <typename Src, typename Dst>
void reorder(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    assert(src_shape == dst_shape);

    if (src_shape.is_contiguous() && dst_shape.is_contiguous() &&
        src.data() == dst.data() && src_shape.offset() == dst_shape.offset())
        return;

    if (src.shape().is_tail(src_shape) && src_shape.is_contiguous() && dst_shape.is_contiguous()) {
        gpgpu::dnn::copy(src.size(), src.data(), src_shape.offset(),
                         dst.size(), dst.data(), dst_shape.offset());
    } else {
        gpgpu::dnn::copy(src_shape.size(), src_shape.extents(),
                         src.data(), src_shape.offset(), src_shape.strides(),
                         dst.data(), dst_shape.offset(), dst_shape.strides());
    }
}
} // namespace detail

template <typename T>
inline void reorder(const DevTensor<T>& src, const Shape& src_shape, DevTensor<T>& dst, const Shape& dst_shape) {
    detail::reorder(src, src_shape, dst, dst_shape);
}

template <typename T>
inline void reorder(const DevTensor<T>& src, const Shape& src_shape, DevTensor<T>& dst) {
    dst.resize(src_shape);
    reorder(src, src_shape, dst, dst.shape());
}

template <typename T>
inline void reorder(const DevTensorView<T>& src, DevTensorView<T>& dst) {
    detail::reorder(src, src.shape(), dst, dst.shape());
}

template <typename T>
inline void reorder(const DevTensorView<T>& src, DevTensorView<T>&& dst) {
    detail::reorder(src, src.shape(), dst, dst.shape());
}

template <typename T>
inline void reorder(const DevTensorView<T>& src, DevTensor<T>& dst) {
    dst.resize(src.shape());
    detail::reorder(src, src.shape(), dst, dst.shape());
}

template <typename T>
inline void reorder(const DevTensor<T>& src, DevTensorView<T>& dst) {
    detail::reorder(src, src.shape(), dst, dst.shape());
}

template <typename T>
inline void reorder(const DevTensor<T>& src, DevTensorView<T>&& dst) {
    detail::reorder(src, src.shape(), dst, dst.shape());
}

template <typename T>
inline void flat_copy(const DevTensor<T>& src, DevTensor<T>& dst) {
    assert(src.size() == dst.size());
    src.copyToAsync(dst);
}

//==-------------------------------------------------------------------------
// DevTensor binary transformations
//==-------------------------------------------------------------------------

template <typename T> struct is_relop                        : public std::false_type {};
template <typename T> struct is_relop<xfn::equal_to<T>>      : public std::true_type {};
template <typename T> struct is_relop<xfn::not_equal_to<T>>  : public std::true_type {};
template <typename T> struct is_relop<xfn::less<T>>          : public std::true_type {};
template <typename T> struct is_relop<xfn::less_equal<T>>    : public std::true_type {};
template <typename T> struct is_relop<xfn::greater<T>>       : public std::true_type {};
template <typename T> struct is_relop<xfn::greater_equal<T>> : public std::true_type {};

namespace detail {
template <typename T, typename R>
void transform(const std::string& name,
               const Shape& shape_A, const gpgpu::Buffer<T>& data_A,
               const Shape& shape_B, const gpgpu::Buffer<T>& data_B,
               const Shape& shape_C, gpgpu::Buffer<R>& data_C)
{
    if (shape_A.is_contiguous() && shape_B.is_contiguous() && shape_C.is_contiguous()) {
        if (shape_A.is_tail(shape_B) || shape_B.is_tail(shape_A)) {
            gpgpu::dnn::transform(name,
                                  shape_A.size(), data_A, shape_A.offset(),
                                  shape_B.size(), data_B, shape_B.offset(),
                                  data_C, shape_C.offset());
            return;
        }

        int axis = shape_A.find_channel_axis(shape_B);
        if (axis != -1) {
            size_t m = shape_A.partial_size(0, axis+1);
            size_t n = shape_A.size() / m;
            gpgpu::dnn::transform(name, m, n, shape_B.size(),
                                  data_A, shape_A.offset(),
                                  data_B, shape_B.offset(),
                                  data_C, shape_C.offset());
            return;
        }
    }

    auto sA = shape_A.broadcast(shape_C);
    auto sB = shape_B.broadcast(shape_C);
    gpgpu::dnn::transform(name, shape_C.size(), shape_C.extents(),
                          data_A, sA.offset(), sA.strides(),
                          data_B, sB.offset(), sB.strides(),
                          data_C, shape_C.offset(), shape_C.strides());
}

template <typename TensorA, typename TensorB, typename R>
DevTensor<R>& transform(const std::string& name, const TensorA& A, const TensorB& B, DevTensor<R>& C) {
    C.resize(Shape::broadcast(A, B));
    transform(name, A.shape(), A.data(), B.shape(), B.data(), C.shape(), C.data());
    return C;
}

template <typename TensorA, typename TensorB, typename R>
DevTensorView<R>& transform(const std::string& name, const TensorA& A, const TensorB& B, DevTensorView<R>& C) {
    if (C.shape() != Shape::broadcast(A, B))
        throw shape_error("incompatible shape");
    transform(name, A.shape(), A.data(), B.shape(), B.data(), C.shape(), C.data());
    return C;
}
} // namespace detail

template <typename T, typename Fn>
inline DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensor<T>& A, const DevTensor<T>& B,
    DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>>& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensor<T>& A, const DevTensor<T>& B,
    DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensor<T>& A, const DevTensor<T>& B,
    DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensorView<T>& A, const DevTensor<T>& B,
    DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>>& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensorView<T>& A, const DevTensor<T>& B,
    DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensorView<T>& A, const DevTensor<T>& B,
    DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensor<T>& A, const DevTensorView<T>& B,
    DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>>& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensor<T>& A, const DevTensorView<T>& B,
    DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensor<T>& A, const DevTensorView<T>& B,
    DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensorView<T>& A, const DevTensorView<T>& B,
    DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>>& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensorView<T>& A, const DevTensorView<T>& B,
    DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&
transformTo(const DevTensorView<T>& A, const DevTensorView<T>& B,
    DevTensorView<std::conditional_t<is_relop<Fn>::value, bool, T>>&& C, Fn)
{
    return detail::transform(Fn::name, A, B, C);
}

template <typename T, typename Fn>
inline auto transform(const DevTensor<T>& A, const DevTensor<T>& B, Fn fn) {
    DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>> C;
    transformTo(A, B, C, fn);
    return C;
}

template <typename T, typename Fn>
std::enable_if_t<!is_relop<Fn>::value, DevTensor<T>>
inline transform(DevTensor<T>&& A, const DevTensor<T>& B, Fn fn) {
    if (A.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, A, fn));
    else
        return transform(A, B, fn);
}

template <typename T, typename Fn>
std::enable_if_t<!is_relop<Fn>::value, DevTensor<T>>
inline transform(const DevTensor<T>& A, DevTensor<T>&& B, Fn fn) {
    if (B.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, B, fn));
    else
        return transform(A, B, fn);
}

template <typename T, typename Fn>
std::enable_if_t<!is_relop<Fn>::value, DevTensor<T>>
inline transform(DevTensor<T>&& A, DevTensor<T>&& B, Fn fn) {
    Shape final_shape = Shape::broadcast(A, B);
    if (A.shape() == final_shape)
        return std::move(transformTo(A, B, A, fn));
    else if (B.shape() == final_shape)
        return std::move(transformTo(A, B, B, fn));
    else
        return transform(A, B, fn);
}

template <typename T, typename Fn>
inline auto transform(const DevTensorView<T>& A, const DevTensor<T>& B, Fn fn) {
    DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>> C;
    transformTo(A, B, C, fn);
    return C;
}

template <typename T, typename Fn>
std::enable_if_t<!is_relop<Fn>::value, DevTensor<T>>
inline transform(const DevTensorView<T>& A, DevTensor<T>&& B, Fn fn) {
    if (B.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, B, fn));
    else
        return transform(A, B, fn);
}

template <typename T, typename Fn>
inline auto transform(const DevTensor<T>& A, const DevTensorView<T>& B, Fn fn) {
    DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>> C;
    transformTo(A, B, C, fn);
    return C;
}

template <typename T, typename Fn>
std::enable_if_t<!is_relop<Fn>::value, DevTensor<T>>
inline transform(DevTensor<T>&& A, const DevTensorView<T>& B, Fn fn) {
    if (A.shape() == Shape::broadcast(A, B))
        return std::move(transformTo(A, B, A, fn));
    else
        return transform(A, B, fn);
}

template <typename T, typename Fn>
inline auto transform(const DevTensorView<T>& A, const DevTensorView<T>& B, Fn fn) {
    DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>> C;
    transformTo(A, B, C, fn);
    return C;
}

template <typename T, typename Fn>
void transformChannel(
    const DevTensor<T>& A, const DevTensor<T>& B,
    DevTensor<std::conditional_t<is_relop<Fn>::value, bool, T>>& C,
    size_t axis, Fn)
{
    assert(B.is_vector() || A.shape().find_channel_axis(B.shape()) == axis);
    assert(axis < A.rank());
    assert(A.extent(axis) == B.size());
    assert(C.shape() == A.shape());

    size_t m = A.shape().partial_size(0, axis+1);
    size_t n = A.size() / m;
    gpgpu::dnn::transform(Fn::name, m, n, B.size(), A.data(), 0, B.data(), 0, C.data(), 0);
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
void transpose(const DevTensor<T>& src, DevTensor<T>& dst, const std::vector<size_t>& perm) {
    Shape shape = src.shape().transpose(perm);
    dst.resize(shape);

    if (shape.rank() == 2 && !shape.is_contiguous()) {
        gpgpu::blas::omatcopy(gpgpu::blas::Layout::RowMajor,
                              gpgpu::blas::Transpose::Trans,
                              src.extent(0), src.extent(1),
                              T(1), src.data(), src.stride(0),
                              dst.data(), dst.stride(0));
    } else {
        reorder(src, shape, dst);
    }
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
DevTensorView<T> DevTensor<T>::slice(const std::vector<SliceDim>& dims) const {
    return DevTensorView<T>(shape().slice(dims), *this);
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
