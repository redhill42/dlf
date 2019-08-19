#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor reduction implementation
//==-------------------------------------------------------------------------

namespace detail {
template <typename T, typename Reduction>
void reduce(const Shape& x_shape, const T* x_data,
            const Shape& y_shape, T* y_data,
            size_t n, Reduction reduction)
{
    if (x_shape.is_contiguous()) {
        tbb::parallel_for(tbb::blocked_range<int>(0, x_shape.size()/n, 1), [&](auto r) {
            for (int i = r.begin(); i < r.end(); i++) {
                auto px = x_data + i*n + x_shape.offset();
                T acc = Reduction::identity;
                for (int j = 0; j < n; ++j, ++px)
                    acc = reduction(acc, *px);
                y_data[i] = Reduction::post(acc, n);
            }
        });
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0, x_shape.size()/n, 1), [&](auto r) {
            for (int i = r.begin(); i < r.end(); i++) {
                auto px = const_shaped_iterator<T>(x_shape, x_data, i*n);
                T acc = Reduction::identity;
                for (int j = 0; j < n; ++j, ++px)
                    acc = reduction(acc, *px);
                y_data[i] = Reduction::post(acc, n);
            }
        });
    }
}

template <typename T, typename Reduction>
void reduce(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
            const Shape& y_shape, gpgpu::Buffer<T>& y_data,
            size_t n, Reduction)
{
    gpgpu::dnn::reduce(Reduction::name, x_shape.size()/n, n,
                       x_shape.extents(), x_shape.strides(),
                       x_data, x_shape.offset(), y_data, 0);

}

template <typename TensorT, typename Reduction>
void reduce(const TensorT& X, tensor_type<TensorT>& Y, Reduction reduction,
            std::vector<int>&& axes, bool keepdims)
{
    auto rank = X.rank();

    // normalize axes
    for (auto& a : axes) {
        if (a < 0) a += rank;
        if (a < 0 || a >= rank)
            throw shape_error("reduce: axes has incorrect value");
    }

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
    reduce(x_shape, X.data(), y_shape, Y.data(), n, reduction);
}
} // namespace detail

//==-------------------------------------------------------------------------
// Uniform reduction operations
//==-------------------------------------------------------------------------

template <typename TensorT, typename Reduction>
inline enable_if_tensor<TensorT, void> reduce(
    const TensorT& X, tensor_type<TensorT>& Y, Reduction reduction,
    std::vector<int> axes = {}, bool keepdims = false)
{
    detail::reduce(X, Y, reduction, std::move(axes), keepdims);
}

template <typename TensorT, typename Reduction>
inline enable_if_tensor<TensorT> reduce(
    const TensorT& X, Reduction reduction,
    std::vector<int> axes = {}, bool keepdims = false)
{
    tensor_type<TensorT> Y;
    reduce(X, Y, reduction, std::move(axes), keepdims);
    return Y;
}

#define DEFINE_REDUCE_OP(name)                                              \
template <typename TensorT>                                                 \
inline enable_if_tensor<TensorT, void> name(                                \
    const TensorT& X, tensor_type<TensorT>& Y,                              \
    std::vector<int> axes = {}, bool keepdims = false)                      \
{                                                                           \
    using T = tensor_value_type<TensorT>;                                   \
    reduce(X, Y, xfn::name<T>(), std::move(axes), keepdims);                \
}                                                                           \
                                                                            \
template <typename TensorT>                                                 \
inline enable_if_tensor<TensorT>                                            \
name(const TensorT& X, std::vector<int> axes = {}, bool keepdims = false)   \
{                                                                           \
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

inline Tensor<bool> reduce_all(const Tensor<bool>& X, std::vector<int> axes = {}, bool keepdims = false) {
    return reduce(X, xfn::reduce_all(), std::move(axes), keepdims);
}

inline Tensor<bool> reduce_any(const Tensor<bool>& X, std::vector<int> axes = {}, bool keepdims = false) {
    return reduce(X, xfn::reduce_any(), std::move(axes), keepdims);
}

//==-------------------------------------------------------------------------
// Arg reduction operations
//==-------------------------------------------------------------------------

namespace detail {
inline int norm_axis(const char* name, int axis, size_t rank) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error(cxx::string_concat(name, ": invalid axis"));
    return axis;
}

inline Shape get_reduced_shape(const Shape& x_shape, int axis, bool keepdims) {
    auto y_dims = x_shape.extents();
    if (keepdims) {
        y_dims[axis] = 1;
    } else {
        y_dims.erase(y_dims.begin() + axis);
    }
    return Shape(y_dims);
}

template <typename T, typename Compare>
void arg_reduce(
    const char* name, const Tensor<T>& X, Tensor<int>& Y,
    int axis, bool keepdims, Compare compare)
{
    axis = norm_axis(name, axis, X.rank());
    Y.resize(get_reduced_shape(X.shape(), axis, keepdims));

    auto m = X.shape().partial_size(0, axis);
    auto k = X.extent(axis);
    auto n = X.shape().partial_size(axis+1, X.rank());

    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 16, 0, n, 64), [=](auto r) {
        for (int i = r.rows().begin(); i < r.rows().end(); i++) {
            for (int j = r.cols().begin(); j < r.cols().end(); j++) {
                auto px = x_buffer + i * k * n + j;
                T acc = *px;
                int index = 0;
                for (int ik = 1; ik < k; ik++) {
                    if (compare(px[ik*n], acc)) {
                        acc = px[ik*n];
                        index = ik;
                    }
                }
                y_buffer[i * n + j] = index;
            }
        }
    });
}
} // namespace detail

template <typename T>
void argmax(const Tensor<T>& X, Tensor<int>& Y, int axis, bool keepdims = true) {
    detail::arg_reduce("argmax", X, Y, axis, keepdims, std::greater<>());
}

template <typename T>
void argmin(const Tensor<T>& X, Tensor<int>& Y, int axis,bool keepdims = true) {
    detail:: arg_reduce("argmin", X, Y, axis, keepdims, std::less<>());
}

template <typename T>
void argmax(const DevTensor<T>& X, DevTensor<int>& Y, int axis, bool keepdims = true) {
    axis = detail::norm_axis("argmax", axis, X.rank());
    Y.resize(detail::get_reduced_shape(X.shape(), axis, keepdims));

    auto m = X.shape().partial_size(0, axis);
    auto k = X.extent(axis);
    auto n = X.shape().partial_size(axis+1, X.rank());
    gpgpu::dnn::argmax(m, k, n, X.data(), Y.data());
}

template <typename T>
void argmin(const DevTensor<T>& X, DevTensor<int>& Y, int axis, bool keepdims = true) {
    axis = detail::norm_axis("argmin", axis, X.rank());
    Y.resize(detail::get_reduced_shape(X.shape(), axis, keepdims));

    auto m = X.shape().partial_size(0, axis);
    auto k = X.extent(axis);
    auto n = X.shape().partial_size(axis+1, X.rank());
    gpgpu::dnn::argmin(m, k, n, X.data(), Y.data());
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, tensor_type<TensorT, int>>
argmax(const TensorT& X, int axis, bool keepdims = true) {
    tensor_type<TensorT, int> Y{};
    argmax(X, Y, axis, keepdims);
    return Y;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, tensor_type<TensorT, int>>
argmin(const TensorT& X, int axis, bool keepdims = true) {
    tensor_type<TensorT, int> Y{};
    argmin(X, Y, axis, keepdims);
    return Y;
}

} // namespace dlf
