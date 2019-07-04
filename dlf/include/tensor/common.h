#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor unary operations
//==-------------------------------------------------------------------------

namespace detail {
template <typename TensorT>
struct tensor_traits_impl {
    static constexpr bool is_tensor = false;
};

template <typename T>
struct tensor_traits_impl<Tensor<T>> {
    static constexpr bool is_tensor = true;
    using type = Tensor<T>;
    using reference = type&;
    using const_reference = const type&;
    using value_type = T;
    using value_reference = value_type&;
    using const_value_reference = const value_type&;
};

template <typename T>
struct tensor_traits_impl<DevTensor<T>> {
    static constexpr bool is_tensor = true;
    using type = DevTensor<T>;
    using reference = type&;
    using const_reference = const type&;
    using value_type = T;
    using value_reference = value_type&;
    using const_value_reference = const value_type&;
};
} // namespace detail

template <typename TensorT>
struct tensor_traits : detail::tensor_traits_impl<std::decay_t<TensorT>> {};

template <typename TensorT, typename T = void>
using enable_if_tensor = std::enable_if_t<tensor_traits<TensorT>::is_tensor, T>;

template <typename TensorT>
using tensor_type = typename tensor_traits<TensorT>::type;

template <typename TensorT>
using tensor_value_type = typename tensor_traits<TensorT>::value_type;

#define DEFINE_UNARY_OPERATOR(name, op) \
template <typename TensorT, enable_if_tensor<TensorT, int> = 0> \
inline tensor_type<TensorT> name(TensorT&& x) { \
    using T = tensor_value_type<TensorT>; \
    return ::dlf::transform(std::forward<TensorT>(x), ::dlf::xfn::op<T>()); \
}

DEFINE_UNARY_OPERATOR(abs, abs)
DEFINE_UNARY_OPERATOR(operator-, negate)
DEFINE_UNARY_OPERATOR(sign, sign)
DEFINE_UNARY_OPERATOR(reciprocal, reciprocal)
DEFINE_UNARY_OPERATOR(floor, floor)
DEFINE_UNARY_OPERATOR(ceil, ceil)
DEFINE_UNARY_OPERATOR(round, round)
DEFINE_UNARY_OPERATOR(sqrt, sqrt)
DEFINE_UNARY_OPERATOR(exp, exp)
DEFINE_UNARY_OPERATOR(log, log)
DEFINE_UNARY_OPERATOR(sin, sin)
DEFINE_UNARY_OPERATOR(cos, cos)
DEFINE_UNARY_OPERATOR(tan, tan)
DEFINE_UNARY_OPERATOR(asin, asin)
DEFINE_UNARY_OPERATOR(acos, acos)
DEFINE_UNARY_OPERATOR(atan, atan)
DEFINE_UNARY_OPERATOR(sinh, sinh)
DEFINE_UNARY_OPERATOR(cosh, cosh)
DEFINE_UNARY_OPERATOR(tanh, tanh)
DEFINE_UNARY_OPERATOR(asinh, asinh)
DEFINE_UNARY_OPERATOR(acosh, acosh)
DEFINE_UNARY_OPERATOR(atanh, atanh)
DEFINE_UNARY_OPERATOR(erf, erf)
DEFINE_UNARY_OPERATOR(sigmoid, sigmoid)
#undef DEFINE_UNARY_OPERATOR

//==-------------------------------------------------------------------------
// Tensor shape operations
//==-------------------------------------------------------------------------

template <typename TensorT, enable_if_tensor<TensorT, int> = 0>
inline tensor_type<TensorT> reshape(TensorT&& tensor, const std::vector<size_t>& newshape) {
    tensor_type<TensorT> ret = std::forward<TensorT>(tensor);
    if (!ret.reshape(newshape))
        throw shape_error("cannot reshape to given shape");
    return ret;
}

template <typename T>
inline void reshape(const Tensor<T>& src, Tensor<T>& dst) {
    if (src.size() != dst.size())
        throw shape_error("cannot reshape to destination tensor");
    if (src.data() != dst.data())
        std::copy(src.begin(), src.end(), dst.begin());
}

template <typename T>
inline void reshape(const DevTensor<T>& src, DevTensor<T>& dst) {
    if (src.size() != dst.size())
        throw shape_error("cannot reshape to destination tensor");
    if (src.data() != dst.data())
        src.data().copyToAsync(gpgpu::current::queue(), dst.data(), dst.size());
}

template <typename TensorT, enable_if_tensor<TensorT, int> = 0>
inline tensor_type<TensorT> flatten(TensorT&& tensor, size_t axis) {
    if (axis > tensor.shape().rank())
        throw std::logic_error("flatten: invalid axis");

    auto dims = tensor.shape().extents();
    size_t rows = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
    size_t cols = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());
    return reshape(std::forward<TensorT>(tensor), {rows, cols});
}

} // namespace dlf
