#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor reduction implementation
//==-------------------------------------------------------------------------

namespace detail {
template <typename T, typename Reduction, typename Post>
void reduce(const Shape& x_shape, const T* x_data,
            const Shape& y_shape, T* y_data,
            size_t n, Reduction reduction, Post post)
{
    if (x_shape.is_contiguous()) {
        tbb::parallel_for(tbb::blocked_range<int>(0, x_shape.size()/n, 1), [&](auto r) {
            for (int i = r.begin(); i < r.end(); i++) {
                auto px = x_data + i*n + x_shape.offset();
                T acc = Reduction::identity;
                for (int j = 0; j < n; ++j, ++px)
                    acc = reduction(acc, *px);
                y_data[i] = post(acc, n);
            }
        });
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0, x_shape.size()/n, 1), [&](auto r) {
            for (int i = r.begin(); i < r.end(); i++) {
                auto px = const_shaped_iterator<T>(x_shape, x_data, i*n);
                T acc = Reduction::identity;
                for (int j = 0; j < n; ++j, ++px)
                    acc = reduction(acc, *px);
                y_data[i] = post(acc, n);
            }
        });
    }
}

template <typename T, typename Reduction, typename Post>
void reduce(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
            const Shape& y_shape, gpgpu::Buffer<T>& y_data,
            size_t n, Reduction, Post)
{
    gpgpu::dnn::reduce(Reduction::name, x_shape.size()/n, n,
                       x_shape.extents(), x_shape.strides(),
                       x_data, x_shape.offset(), y_data, 0);
}

template <typename TensorT, typename Reduction, typename Post>
void reduce(const TensorT& X, tensor_type<TensorT>& Y,
            Reduction reduction, Post post,
            std::vector<int>&& axes, bool keepdims)
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
    if (output_dims.empty()) {
        output_dims.push_back(1);
    }

    auto x_shape = X.shape().transpose(transpose_perm);
    auto y_shape = Shape(output_dims);

    Y.resize(y_shape);
    reduce(x_shape, X.data(), y_shape, Y.data(), n, reduction, post);
}
} // namespace detail

//==-------------------------------------------------------------------------
// Uniform reduction operations
//==-------------------------------------------------------------------------

template <typename TensorT, typename Reduction, typename Post>
inline enable_if_tensor<TensorT, void> reduce(
    const TensorT& X, tensor_type<TensorT>& Y,
    Reduction reduction, Post post,
    std::vector<int> axes = {}, bool keepdims = false)
{
    detail::reduce(X, Y, reduction, post, std::move(axes), keepdims);
}

template <typename TensorT, typename Reduction, typename Post>
inline enable_if_tensor<TensorT> reduce(
    const TensorT& X, Reduction reduction, Post post,
    std::vector<int> axes = {}, bool keepdims = false)
{
    tensor_type<TensorT> Y;
    reduce(X, Y, reduction, post, std::move(axes), keepdims);
    return Y;
}

#define DEFINE_REDUCE_OP(name)                                              \
template <typename TensorT>                                                 \
inline enable_if_tensor<TensorT, void> name(                                \
    const TensorT& X, tensor_type<TensorT>& Y,                              \
    std::vector<int> axes = {}, bool keepdims = false)                      \
{                                                                           \
    using T = tensor_value_type<TensorT>;                                   \
    using Reduction = xfn::name<T>;                                         \
    using Post = typename Reduction::Post;                                  \
    reduce(X, Y, Reduction{}, Post{}, std::move(axes), keepdims);           \
}                                                                           \
                                                                            \
template <typename TensorT>                                                 \
inline enable_if_tensor<TensorT>                                            \
name(const TensorT& X, std::vector<int> axes = {}, bool keepdims = false) { \
    tensor_type<TensorT> Y;                                                 \
    name(X, Y, std::move(axes), keepdims);                                  \
    return Y;                                                               \
}

DEFINE_REDUCE_OP(reduce_max)
DEFINE_REDUCE_OP(reduce_min)
DEFINE_REDUCE_OP(reduce_sum)
DEFINE_REDUCE_OP(reduce_mean)
DEFINE_REDUCE_OP(reduce_sum_square)
DEFINE_REDUCE_OP(reduce_log_sum)
DEFINE_REDUCE_OP(reduce_log_sum_exp)
DEFINE_REDUCE_OP(reduce_prod)
DEFINE_REDUCE_OP(reduce_l1)
DEFINE_REDUCE_OP(reduce_l2)
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

} // namespace dlf
