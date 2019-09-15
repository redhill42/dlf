#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor reduction implementation
//==-------------------------------------------------------------------------

namespace detail {
template <typename T, typename IteratorX, typename IteratorY, typename Reduction, typename Post>
void reduce(size_t m, size_t n, const T& identity,
            IteratorX x_begin, IteratorY y_begin,
            Reduction reduction, Post post)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, m, 1), [&](auto r) {
        auto px = x_begin + r.begin()*n;
        auto py = y_begin + r.begin();
        for (int i = r.size(); i > 0; --i, ++py) {
            T acc = identity;
            for (int j = 0; j < n; ++j, ++px)
                acc = reduction(acc, *px);
            *py = post(acc, n);
        }
    });
}

template <typename T, typename IteratorX, typename Reduction, typename Post>
inline void reduce(size_t m, size_t n, const T& identity,
                   IteratorX x_begin, const Shape& y_shape, T* y_data,
                   Reduction reduction, Post post)
{
    if (y_shape.is_contiguous()) {
        reduce(m, n, identity, x_begin, y_data + y_shape.offset(), reduction, post);
    } else {
        reduce(m, n, identity, x_begin, shaped_iterator<T>(y_shape, y_data, 0), reduction, post);
    }
}

template <typename T, typename Reduction, typename Post>
inline void reduce(size_t n, const T& identity,
                   const Shape& x_shape, const T* x_data,
                   const Shape& y_shape, T* y_data,
                   Reduction reduction, Post post)
{
    auto m = x_shape.size() / n;
    if (x_shape.is_contiguous()) {
        reduce(m, n, identity, x_data + x_shape.offset(), y_shape, y_data, reduction, post);
    } else {
        reduce(m, n, identity, const_shaped_iterator<T>(x_shape, x_data, 0), y_shape, y_data, reduction, post);
    }
}

template <typename T, typename Reduction, typename Post>
void reduce(size_t n, const T&,
            const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
            const Shape& y_shape, gpgpu::Buffer<T>& y_data,
            Reduction, Post)
{
    gpgpu::dnn::reduce(Reduction::name, x_shape.size()/n, n,
                       x_shape.extents(), x_shape.strides(),
                       x_data, x_shape.offset(),
                       y_shape.extents(), y_shape.strides(),
                       y_data, y_shape.offset());
}

template <typename TensorX, typename TensorY, typename Reduction, typename Post>
void reduce(const TensorX& X, TensorY& Y,
            std::vector<int>&& axes, bool keepdims,
            const tensor_value_type<TensorX>& identity,
            Reduction reduction, Post post)
{
    auto rank = X.rank();
    detail::norm_axes(rank, axes, true);

    std::vector<size_t> output_dims;
    std::vector<size_t> transpose_perm;
    size_t n = 1;

    for (int i = 0; i < rank; i++) {
        // axes empty means reduce all dim
        if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
            output_dims.push_back(X.extent(i));
            transpose_perm.push_back(i);
        } else if (keepdims) {
            output_dims.push_back(1);
        }
    }
    for (int i = 0; i < rank; i++) {
        if (axes.empty() || std::find(axes.begin(), axes.end(), i) != axes.end()) {
            transpose_perm.push_back(i);
            n *= X.extent(i);
        }
    }

    auto x_shape = X.shape().transpose(transpose_perm);
    Y.resize(Shape(output_dims));
    reduce(n, identity, x_shape, X.data(), Y.shape(), Y.data(), reduction, post);
}
} // namespace detail

//==-------------------------------------------------------------------------
// Uniform reduction operations
//==-------------------------------------------------------------------------

template <typename TensorX, typename TensorY, typename Reduction, typename Post>
std::enable_if_t<
    is_cpu_tensor<TensorX>::value &&
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    !std::is_const<std::remove_reference_t<TensorY>>::value>
inline reduce(
    const TensorX& X, TensorY&& Y,
    std::vector<int> axes, bool keepdims,
    const tensor_value_type<TensorX>& identity,
    Reduction reduction, Post post)
{
    detail::reduce(X, Y, std::move(axes), keepdims, identity, reduction, post);
}

template <typename TensorT, typename Reduction, typename Post>
std::enable_if_t<is_cpu_tensor<TensorT>::value, tensor_type<TensorT>>
inline reduce(
    const TensorT& X, std::vector<int> axes, bool keepdims,
    const tensor_value_type<TensorT>& identity,
    Reduction reduction, Post post)
{
    tensor_type<TensorT> Y{};
    detail::reduce(X, Y, std::move(axes), keepdims, identity, reduction, post);
    return Y;
}

template <typename Reducer, typename Post = typename Reducer::Post,
          typename TensorX, typename TensorY>
std::enable_if_t<
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    !std::is_const<std::remove_reference_t<TensorY>>::value>
inline reduce(const TensorX& X, TensorY&& Y, std::vector<int> axes = {}, bool keepdims = false) {
    detail::reduce(X, Y, std::move(axes), keepdims, Reducer::identity(), Reducer{}, Post{});
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
// Arg reduction operations
//==-------------------------------------------------------------------------

namespace detail {
template <typename T, typename Compare>
void arg_reduce(const TensorView<T>& X, Tensor<int>& Y, const char*, Compare compare) {
    auto k = X.shape().extent(-1);
    auto strideK = X.shape().stride(-1);
    auto n = X.size() / k;
    auto x_buffer = X.data();
    auto x_shape  = X.shape();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, n, std::max(size_t(1), GRAINSIZE/k)), [=](auto r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            auto px = x_buffer + x_shape.linear_offset(i*k);
            T acc = *px;
            int index = 0;
            for (int ik = 1; ik < k; ++ik) {
                px += strideK;
                if (compare(*px, acc)) {
                    acc = *px;
                    index = ik;
                }
            }
            y_buffer[i] = index;
        }
    });
}

template <typename T, typename Compare>
void arg_reduce(const DevTensorView<T>& X, DevTensor<int>& Y, const char* name, Compare) {
    auto k = X.shape().extent(-1);
    auto n = X.size() / k;
    gpgpu::dnn::arg_reduce(
        name, n, k,
        X.shape().extents(),
        X.shape().strides(),
        X.data(), X.shape().offset(),
        Y.data());
}

template <typename TensorT, typename Compare>
void arg_reduce(
    const TensorT& X, tensor_type<TensorT, int>& Y,
    int axis, bool keepdims, const char* name, Compare compare)
{
    norm_axis(X.rank(), axis);

    auto y_dims = X.shape().extents();
    if (keepdims) {
        y_dims[axis] = 1;
    } else {
        y_dims.erase(y_dims.begin() + axis);
    }
    Y.resize(Shape{y_dims});

    arg_reduce(moveaxis(X, axis, -1), Y, name, compare);
}
} // namespace detail

template <typename TensorT>
enable_if_tensor<TensorT, void>
inline argmax(const TensorT& X, tensor_type<TensorT, int>& Y, int axis, bool keepdims = true) {
    detail::arg_reduce(X, Y, axis, keepdims, "argmax", std::greater<>());
}

template <typename TensorT>
enable_if_tensor<TensorT, void>
inline argmin(const TensorT& X, tensor_type<TensorT, int>& Y, int axis, bool keepdims = true) {
    detail::arg_reduce(X, Y, axis, keepdims, "argmin", std::less<>());
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT, int>>
argmax(const TensorT& X, int axis, bool keepdims = true) {
    tensor_type<TensorT, int> Y{};
    argmax(X, Y, axis, keepdims);
    return Y;
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

namespace detail {
template <typename T, typename Op>
void scan(int m, int n, bool exclusive, const T& id, Op op,
          const Shape& x_shape, const T* x_data,
          const Shape& y_shape, T* y_data)
{
    const auto grainsize = std::max(1, GRAINSIZE/n);
    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](const auto& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            tbb::parallel_scan(tbb::blocked_range<int>(0, n, GRAINSIZE),
                id,
                [&](const auto& c, auto acc, const bool is_final_scan) {
                    auto px = x_data + x_shape.linear_offset(i*n + c.begin());
                    auto py = y_data + y_shape.linear_offset(i*n + c.begin());
                    auto incX = x_shape.stride(-1);
                    auto incY = y_shape.stride(-1);

                    if (is_final_scan && exclusive) {
                        for (int j = c.size(); j > 0; --j, px += incX, py += incY) {
                            *py = acc;
                            acc = op(acc, *px);
                        }
                    } else if (is_final_scan && !exclusive) {
                        for (int j = c.size(); j > 0; --j, px += incX, py += incY) {
                            acc = op(acc, *px);
                            *py = acc;
                        }
                    } else {
                        for (int j = c.size(); j > 0; --j, px += incX) {
                            acc = op(acc, *px);
                        }
                    }
                    return acc;
                },
                op);
        }
    });
}

template <typename T, typename Op>
void scan(int m, int n, bool exclusive, const T&, Op,
          const Shape& x_shape, const gpgpu::Buffer<T>& x_buffer,
          const Shape& y_shape, gpgpu::Buffer<T>& y_buffer)
{
    gpgpu::dnn::scan(Op::cumulative, m, n, exclusive, x_shape.extents(),
                     x_buffer, x_shape.offset(), x_shape.strides(),
                     y_buffer, y_shape.offset(), y_shape.strides());
}
} // namespace detail

template <typename TensorT, typename TensorR, typename Op>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
scan(const TensorT& X, TensorR&& Y,
     int axis, bool exclusive, bool reverse,
     const tensor_value_type<TensorT>& id, Op op)
{
    detail::norm_axis(X.rank(), axis);
    int n = X.extent(axis);
    int m = X.size() / n;
    Y.resize(X.shape());

    auto x_t = moveaxis(X, axis, -1);
    auto y_t = moveaxis(Y, axis, -1);
    if (reverse) {
        x_t = flip(x_t, -1);
        y_t = flip(y_t, -1);
    }
    detail::scan(m, n, exclusive, id, op,
                 x_t.shape(), x_t.data(),
                 y_t.shape(), y_t.data());
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

} // namespace dlf
