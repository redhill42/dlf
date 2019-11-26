#pragma once

#include "./reduce_detail.h"

namespace dlf {

//==-------------------------------------------------------------------------
// CPU only reduction using map/reduce/final functions
//==-------------------------------------------------------------------------

template <typename TensorT, typename TensorR, typename U,
          typename Map, typename Reduce, typename Final>
std::enable_if_t<
    is_cpu_tensor<TensorT>::value &&
    is_same_tensor<TensorT, TensorR>::value &&
    std::is_convertible<U, tensor_value_type<TensorR>>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline reduce(const TensorT& X, TensorR&& Y,
              std::vector<int> axes, bool keepdims,
              const U& identity, Map f, Reduce g, Final h)
{
    detail::reduce(X, Y, std::move(axes), keepdims, identity, f, g, h);
}

template <typename TensorT, typename TensorR, typename U, typename Map, typename Reduce>
std::enable_if_t<
    is_cpu_tensor<TensorT>::value &&
    is_same_tensor<TensorT, TensorR>::value &&
    std::is_convertible<U, tensor_value_type<TensorR>>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline reduce(const TensorT& X, TensorR&& Y,
              std::vector<int> axes, bool keepdims,
              const U& identity, Map f, Reduce g)
{
    using R = tensor_value_type<TensorR>;
    reduce(X, Y, std::move(axes), keepdims, identity, f, g, xfn::post_reduce_identity<R>());
}

template <typename TensorT, typename U, typename Map, typename Reduce, typename Final>
std::enable_if_t<
    is_cpu_tensor<TensorT>::value &&
    std::is_convertible<U, cxx::invoke_result_t<Map, tensor_value_type<TensorT>>>::value,
    tensor_type<TensorT, cxx::invoke_result_t<Map, tensor_value_type<TensorT>>>>
inline reduce(const TensorT& X, std::vector<int> axes, bool keepdims,
              const U& identity, Map f, Reduce g, Final h)
{
    tensor_type<TensorT, cxx::invoke_result_t<Map, tensor_value_type<TensorT>>> Y{};
    reduce(X, Y, std::move(axes), keepdims, identity, f, g, h);
    return Y;
}

template <typename TensorT, typename U, typename Map, typename Reduce>
std::enable_if_t<
    is_cpu_tensor<TensorT>::value &&
    std::is_convertible<U, cxx::invoke_result_t<Map, tensor_value_type<TensorT>>>::value,
    tensor_type<TensorT, cxx::invoke_result_t<Map, tensor_value_type<TensorT>>>>
inline reduce(const TensorT& X, std::vector<int> axes, bool keepdims,
              const U& identity, Map f, Reduce g)
{
    using R = cxx::invoke_result_t<Map, tensor_value_type<TensorT>>;
    return reduce(X, std::move(axes), keepdims, identity, f, g, xfn::post_reduce_identity<R>());
}

//==-------------------------------------------------------------------------
// Uniform reduction using predefined reducer
//==-------------------------------------------------------------------------

template <typename Reducer, typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline reduce(const TensorT& X, TensorR&& Y, std::vector<int> axes = {}, bool keepdims = false) {
    detail::reduce<Reducer>(X, Y, std::move(axes), keepdims);
}

template <typename Reducer, typename TensorT>
enable_if_tensor<TensorT>
inline reduce(const TensorT& X, std::vector<int> axes = {}, bool keepdims = false) {
    tensor_type<TensorT> Y{};
    reduce<Reducer>(X, Y, std::move(axes), keepdims);
    return Y;
}

#define DEFINE_REDUCE_OP(name)                                              \
template <typename TensorX, typename TensorY>                               \
std::enable_if_t<                                                           \
    is_exactly_same_tensor<TensorX, TensorY>::value &&                      \
    !std::is_const<std::remove_reference_t<TensorY>>::value>                \
inline name(const TensorX& X, TensorY&& Y,                                  \
            std::vector<int> axes = {}, bool keepdims = false)              \
{                                                                           \
    using Reducer = xfn::name<tensor_value_type<TensorX>>;                  \
    reduce<Reducer>(X, std::forward<TensorY>(Y), std::move(axes), keepdims);\
}                                                                           \
                                                                            \
template <typename TensorT>                                                 \
inline enable_if_tensor<TensorT>                                            \
name(const TensorT& X, std::vector<int> axes = {}, bool keepdims = false) { \
    tensor_type<TensorT> Y;                                                 \
    name(X, Y, std::move(axes), keepdims);                                  \
    return Y;                                                               \
}                                                                           \
                                                                            \
template <typename TensorT>                                                 \
inline enable_if_tensor<TensorT>                                            \
name(const TensorT& X, std::initializer_list<int> axes, bool keepdims = false) { \
    return name(X, std::vector<int>(axes), keepdims);                       \
}                                                                           \
                                                                            \
template <typename TensorT>                                                 \
inline enable_if_tensor<TensorT>                                            \
name(const TensorT& X, int axis, bool keepdims = false) {                   \
    return name(X, std::vector<int>{axis}, keepdims);                       \
}

DEFINE_REDUCE_OP(reduce_max)
DEFINE_REDUCE_OP(reduce_min)
DEFINE_REDUCE_OP(reduce_amax)
DEFINE_REDUCE_OP(reduce_amin)
DEFINE_REDUCE_OP(reduce_sum)
DEFINE_REDUCE_OP(reduce_asum)
DEFINE_REDUCE_OP(reduce_mean)
DEFINE_REDUCE_OP(reduce_sum_square)
DEFINE_REDUCE_OP(reduce_nrm2)
DEFINE_REDUCE_OP(reduce_log_sum)
DEFINE_REDUCE_OP(reduce_sum_exp)
DEFINE_REDUCE_OP(reduce_log_sum_exp)
DEFINE_REDUCE_OP(reduce_prod)
#undef DEFINE_REDUCE_OP

template <typename TensorT>
inline enable_if_tensor<TensorT>
reduce_var(const TensorT& X, std::vector<int> axes = {}, bool keepdims = false) {
    return reduce_mean(square(X), axes, keepdims) -
           square(reduce_mean(X, axes, keepdims));
}

template <typename TensorT>
inline enable_if_tensor<TensorT>
reduce_std(const TensorT& X, std::vector<int> axes = {}, bool keepdims = false) {
    return sqrt(reduce_var(X, axes, keepdims));
}

//==-------------------------------------------------------------------------

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_cpu_tensor<TensorT>::value &&
    is_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline count(const TensorT& X, TensorR&& Y, const tensor_value_type<TensorT>& value,
            std::vector<int> axes = {}, bool keepdims = false)
{
    using R = tensor_value_type<TensorR>;
    detail::reduce(X, Y, std::move(axes), keepdims, xfn::zero<R>(),
                   [value](auto x){ return x==value ? xfn::one<R>() : xfn::zero<R>(); },
                   xfn::plus<R>(), xfn::post_reduce_identity<R>());
}

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_gpu_tensor<TensorT>::value &&
    is_same_tensor<TensorR, tensor_type<TensorT, int>>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline count(const TensorT& X, TensorR&& Y, const tensor_value_type<TensorT>& value,
             std::vector<int> axes = {}, bool keepdims = false)
{
    int m, n;
    auto x_shape = detail::prepare_reduce(X, Y, std::move(axes), keepdims, &m, &n);
    gpgpu::dnn::count(m, n, value,
                      x_shape.extents(), x_shape.strides(),
                      X.data(), x_shape.offset(),
                      Y.shape().extents(), Y.shape().strides(),
                      Y.data(), Y.shape().offset());
}

template <typename R = int, typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT, R>>
inline count(const TensorT& X, const tensor_value_type<TensorT>& value,
             std::vector<int> axes = {}, bool keepdims = false) {
    tensor_type<TensorT, R> Y{};
    count(X, Y, value, std::move(axes), keepdims);
    return Y;
}

template <typename R = int, typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT, R>>
inline count(const TensorT& X, const tensor_value_type<TensorT>& value,
             std::initializer_list<int> axes, bool keepdims = false) {
    return count(X, value, std::vector<int>(axes), keepdims);
}

template <typename R = int, typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT, R>>
inline count(const TensorT& X, const tensor_value_type<TensorT>& value,
             int axis, bool keepdims = false) {
    return count(X, value, std::vector<int>{axis}, keepdims);
}

//==-------------------------------------------------------------------------
// Arg reduction operations
//==-------------------------------------------------------------------------

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorR, tensor_type<TensorT, int>>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline argmax(const TensorT& X, TensorR&& Y, int axis, bool keepdims = true) {
    detail::arg_reduce(X, Y, axis, keepdims, "argmax", std::greater<>());
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT, int>>
argmax(const TensorT& X, int axis, bool keepdims = true) {
    tensor_type<TensorT, int> Y{};
    argmax(X, Y, axis, keepdims);
    return Y;
}

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorR, tensor_type<TensorT, int>>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline argmin(const TensorT& X, TensorR&& Y, int axis, bool keepdims = true) {
    detail::arg_reduce(X, Y, axis, keepdims, "argmin", std::less<>());
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT, int>>
argmin(const TensorT& X, int axis, bool keepdims = true) {
    tensor_type<TensorT, int> Y{};
    argmin(X, Y, axis, keepdims);
    return Y;
}

//==-------------------------------------------------------------------------
// Scan
//==-------------------------------------------------------------------------

template <typename TensorT, typename TensorR, typename Op>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
scan(const TensorT& X, TensorR&& Y, int axis, bool exclusive, bool reverse,
     const tensor_value_type<TensorT>& id, Op op)
{
    detail::norm_axis(X.rank(), axis);
    int n = X.extent(axis);
    int m = X.size() / n;
    Y.resize(X.shape());

    auto Xt = moveaxis(X, axis, -1);
    auto Yt = moveaxis(Y, axis, -1);
    if (reverse) {
        Xt = flip(Xt, -1);
        Yt = flip(Yt, -1);
    }
    detail::scan(m, n, exclusive, id, op, Xt.shape(), Xt.data(), Yt.shape(), Yt.data());
}

template <typename TensorT, typename Op>
enable_if_tensor<TensorT>
scan(const TensorT& X, int axis, bool exclusive, bool reverse,
     const tensor_value_type<TensorT>& id, Op op)
{
    tensor_type<TensorT> Y{};
    scan(X, Y, axis, exclusive, reverse, id, op);
    return Y;
}

template <typename TensorT, typename Op>
std::enable_if_t<
    is_tensor<TensorT>::value && !is_tensor_view<TensorT>::value &&
    !std::is_lvalue_reference<TensorT>::value,
    tensor_type<TensorT>>
inline scan(TensorT&& X, int axis, bool exclusive, bool reverse,
            const tensor_value_type<TensorT>& id, Op op)
{
    scan(X, X, axis, exclusive, reverse, id, op);
    return std::move(X);
}

template <typename TensorT, typename Op>
enable_if_tensor<TensorT>
inline scan(TensorT&& X, int axis, const tensor_value_type<TensorT>& id, Op op) {
    return scan(std::forward<TensorT>(X), axis, false, false, id, op);
}

/**
 * Returns the cumulative sum of the elements along a given axis.
 */
template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline cumsum(const TensorT& X, TensorR&& Y, int axis = -1,
              bool exclusive = false, bool reverse = false)
{
    using T = tensor_value_type<TensorT>;
    scan(X, std::forward<TensorR>(Y),
         axis, exclusive, reverse,
         xfn::zero<T>(), xfn::plus<>());
}

template <typename TensorT>
enable_if_tensor<TensorT>
inline cumsum(TensorT&& X, int axis = -1, bool exclusive = false, bool reverse = false) {
    using T = tensor_value_type<TensorT>;
    return scan(std::forward<TensorT>(X),
                axis, exclusive, reverse,
                xfn::zero<T>(), xfn::plus<>());
}

/**
 * Returns the cumulative product of elements along a given axis.
 */
template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline cumprod(const TensorT& X, TensorR&& Y, int axis = -1,
               bool exclusive = false, bool reverse = false)
{
    using T = tensor_value_type<TensorT>;
    scan(X, std::forward<TensorR>(Y),
         axis, exclusive, reverse,
         xfn::one<T>(), xfn::multiplies<>());
}

template <typename TensorT>
enable_if_tensor<TensorT>
inline cumprod(TensorT&& X, int axis = -1, bool exclusive = false, bool reverse = false) {
    using T = tensor_value_type<TensorT>;
    return scan(std::forward<TensorT>(X),
                axis, exclusive, reverse,
                xfn::one<T>(), xfn::multiplies<>());
}

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline cummax(const TensorT& X, TensorR&& Y, int axis = -1, bool reverse = false) {
    using T = tensor_value_type<TensorT>;
    scan(X, std::forward<TensorR>(Y),
         axis, false, reverse,
         std::numeric_limits<T>::lowest(), xfn::max<T>());
}

template <typename TensorT>
enable_if_tensor<TensorT>
inline cummax(TensorT&& X, int axis = -1, bool reverse = false) {
    using T = tensor_value_type<TensorT>;
    return scan(std::forward<TensorT>(X),
                axis, false, reverse,
                std::numeric_limits<T>::lowest(), xfn::max<T>());
}

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
inline cummin(const TensorT& X, TensorR&& Y, int axis = -1, bool reverse = false) {
    using T = tensor_value_type<TensorT>;
    scan(X, std::forward<TensorR>(Y),
         axis, false, reverse,
         std::numeric_limits<T>::max(), xfn::min<T>());
}

template <typename TensorT>
enable_if_tensor<TensorT>
inline cummin(TensorT&& X, int axis = -1, bool reverse = false) {
    using T = tensor_value_type<TensorT>;
    return scan(std::forward<TensorT>(X),
                axis, false, reverse,
                std::numeric_limits<T>::max(), xfn::min<T>());
}

//==-------------------------------------------------------------------------

template <typename TensorT>
std::enable_if_t<is_cpu_tensor<TensorT>::value>
nonzero(const TensorT& X, Tensor<int32_t>& Y, bool row_major = false) {
    // Calculate non-zero prefix sum
    auto indices = Tensor<int32_t>({X.size()});
    transformTo(X, reshape(indices, X.shape()), [](auto x){ return x != 0; });
    cumsum(indices, indices);

    // Get number of non-zero values
    auto rank = X.rank();
    auto n    = *(indices.end()-1);
    auto dims = X.shape().extents();

    // Allocate 2D output tensor
    if (row_major)
        Y.resize(rank, n);
    else
        Y.resize(n, rank);

    // Gather non-zero value indices
    map([=, idx = indices.data(), out = Y.data()](auto curr, auto i) {
        auto prev = (i == 0) ? 0 : idx[i - 1];
        if (curr != prev) {
            int temp = i;
            for (int j = static_cast<int>(rank); --j >= 0; ) {
                auto id = row_major ? (n*j + prev) : (prev*rank + j);
                out[id] = temp % dims[j];
                temp /= dims[j];
            }
        }
    })(indices, map_id());
}

template <typename TensorT>
std::enable_if_t<is_gpu_tensor<TensorT>::value>
nonzero(const TensorT& X, DevTensor<int32_t>& Y, bool row_major = false) {
    auto indices_buffer = gpgpu::current::context().getTemporaryBuffer<int32_t>(X.size());
    auto indices_shape  = Shape(X.shape().extents());

    // Calculate non-zero prefix sum
    gpgpu::dnn::scan_nonzero(
        1, X.size(), false, X.shape().extents(),
        X.data(), X.shape().offset(), X.shape().strides(),
        indices_buffer, indices_buffer.offset(), indices_shape.strides());

    // Read number of non-zero values
    int32_t n;
    indices_buffer.read(gpgpu::current::queue(), &n, 1, indices_buffer.offset() + X.size() - 1);

    // Allocate 2D output tensor
    if (row_major)
        Y.resize(X.rank(), n);
    else
        Y.resize(n, X.rank());

    // Gather non-zero value indices
    gpgpu::dnn::gather_indices(X.size(), n, row_major, X.shape().extents(),
                               indices_buffer, indices_buffer.offset(),
                               Y.data(), Y.shape().offset());
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT, int32_t>>
inline nonzero(const TensorT& X, bool row_major = false) {
    tensor_type<TensorT, int32_t> Y{};
    nonzero(X, Y, row_major);
    return Y;
}

} // namespace dlf
