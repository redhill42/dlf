#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor reorder operations
//==-------------------------------------------------------------------------

namespace detail {
#if HAS_MKL
template <typename Src, typename Dst>
std::enable_if_t<
    is_cpu_tensor<Src>::value && is_cpu_tensor<Dst>::value,
    bool>
inline reorder_transpose(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    if (src_shape.rank() != 2)
        return false;
    return cblas::omatcopy2(
        'R', 'T', src_shape.extent(1), src_shape.extent(0),
        tensor_value_type<Src>{1},
        src.data() + src_shape.offset(), src_shape.stride(1), src_shape.stride(0),
        dst.data() + dst_shape.offset(), dst_shape.stride(0), dst_shape.stride(1)) ;
}
#endif

template <typename Src, typename Dst>
std::enable_if_t<is_cpu_tensor<Src>::value && is_cpu_tensor<Dst>::value>
reorder(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    assert(src_shape == dst_shape);
    using T = tensor_value_type<Src>;
    using U = tensor_value_type<Dst>;

    if (src_shape.is_contiguous() && dst_shape.is_contiguous() &&
        src.data() == dst.data() && src_shape.offset() == dst_shape.offset())
        return;

#if HAS_MKL
    if (reorder_transpose(src, src_shape, dst, dst_shape))
        return;
#endif

    if (dst_shape.is_contiguous()) {
        if (src.original_shape().size() == 1) {
            std::fill(dst.data() + dst_shape.offset(),
                      dst.data() + dst_shape.offset() + dst_shape.size(),
                      *src.data());
            return;
        }

        if (src_shape.is_contiguous()) {
            par::copy(src.data() + src_shape.offset(),
                      src.data() + src_shape.offset() + src_shape.size(),
                      dst.data() + dst_shape.offset());
            return;
        }

        par::copy(const_shaped_iterator<T>(src_shape, src.data(), 0),
                  const_shaped_iterator<T>(src_shape, src.data(), src_shape.size()),
                  dst.data() + dst_shape.offset());
    } else {
        if (src.original_shape().size() == 1) {
            std::fill(shaped_iterator<U>(dst_shape, dst.data(), 0),
                      shaped_iterator<U>(dst_shape, dst.data(), dst_shape.size()),
                      *src.data());
            return;
        }

        if (src_shape.is_contiguous()) {
            par::copy(src.data() + src_shape.offset(),
                      src.data() + src_shape.offset() + src_shape.size(),
                      shaped_iterator<U>(dst_shape, dst.data(), 0));
            return;
        }

        par::copy(const_shaped_iterator<T>(src_shape, src.data(), 0),
                  const_shaped_iterator<T>(src_shape, src.data(), src_shape.size()),
                  shaped_iterator<U>(dst_shape, dst.data(), 0));
    }
}

template <typename Src, typename Dst>
std::enable_if_t<is_gpu_tensor<Src>::value && is_gpu_tensor<Dst>::value, bool>
reorder_transpose(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    if (src_shape.rank() != 2)
        return false;
    if (src_shape.stride(0) != 1 || dst_shape.stride(1) != 1)
        return false;
    gpgpu::blas::omatcopy(gpgpu::blas::Layout::RowMajor,
                          gpgpu::blas::Transpose::Trans,
                          src_shape.extent(1),
                          src_shape.extent(0),
                          tensor_value_type<Src>{1},
                          src.data(), src_shape.offset(), src_shape.stride(1),
                          dst.data(), dst_shape.offset(), dst_shape.stride(0));
    return true;
}

template <typename Src, typename Dst>
std::enable_if_t<is_gpu_tensor<Src>::value && is_gpu_tensor<Dst>::value>
reorder(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    assert(src_shape == dst_shape);

    if (src_shape.is_contiguous() && dst_shape.is_contiguous() &&
        src.data() == dst.data() && src_shape.offset() == dst_shape.offset())
        return;

    if (reorder_transpose(src, src_shape, dst, dst_shape))
        return;

    if (src.original_shape().is_tail(src_shape) && dst_shape.is_contiguous()) {
        gpgpu::dnn::copy(src.original_shape().size(), src.data(), src_shape.offset(),
                         dst_shape.size(), dst.data(), dst_shape.offset());
    } else {
        gpgpu::dnn::copy(src_shape.size(), src_shape.extents(),
                         src.data(), src_shape.offset(), src_shape.strides(),
                         dst.data(), dst_shape.offset(), dst_shape.strides());
    }
}
} // namespace detail

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
inline reorder(const TensorT& src, const Shape& src_shape, TensorT& dst, const Shape& dst_shape) {
    detail::reorder(src, src_shape, dst, dst_shape);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
inline reorder(const TensorT& src, const Shape& src_shape, TensorT& dst) {
    dst.resize(src_shape);
    detail::reorder(src, src_shape, dst, dst.shape());
}

template <typename TensorT>
enable_if_tensor<TensorT, void>
inline reorder(const TensorT& src, tensor_type<TensorT>& dst) {
    dst.resize(src.shape());
    detail::reorder(src, src.shape(), dst, dst.shape());
}

template <typename TensorT>
enable_if_tensor<TensorT, void>
inline reorder(const TensorT& src, tensor_view_type<TensorT>& dst) {
    detail::reorder(src, src.shape(), dst, dst.shape());
}

template <typename TensorT>
enable_if_tensor<TensorT, void>
inline reorder(const TensorT& src, tensor_view_type<TensorT>&& dst) {
    detail::reorder(src, src.shape(), dst, dst.shape());
}

//==-------------------------------------------------------------------------
// Reshape operations
//==-------------------------------------------------------------------------

namespace detail {
template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
reshape(const TensorT& X, Shape&& new_shape) {
    if (X.shape().is_contiguous()) {
        return X.view(std::move(new_shape));
    } else {
        tensor_type<TensorT> Y{};
        reorder(X, Y);
        Y.reshape(std::move(new_shape));
        return Y.view();
    }
}
} // namespace detail

template <typename TensorT, typename... Args>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline reshape(TensorT&& X, const std::vector<int>& dims) {
    return detail::reshape(std::forward<TensorT>(X), X.shape().reshape(dims));
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline flatten(TensorT&& X, int axis) {
    return detail::reshape(std::forward<TensorT>(X), X.shape().flatten(axis));
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline flatten(TensorT&& X) {
    return detail::reshape(std::forward<TensorT>(X), X.shape().reshape(-1));
}

template <typename TensorT, typename... Args>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline squeeze(TensorT&& X, Args&&... args) {
    return X.view(X.shape().squeeze(std::forward<Args>(args)...));
}

template <typename TensorT, typename... Args>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline unsqueeze(TensorT&& X, Args&&... args) {
    return X.view(X.shape().unsqueeze(std::forward<Args>(args)...));
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
unsqueeze_left(TensorT&& X, size_t rank) {
    if (X.rank() < rank) {
        std::vector<int> axes(rank - X.rank());
        std::iota(axes.begin(), axes.end(), 0);
        return unsqueeze(std::forward<TensorT>(X), axes);
    }
    return X.view();
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
unsqueeze_right(TensorT&& X, size_t rank) {
    if (X.rank() < rank) {
        std::vector<int> axes(rank - X.rank());
        std::iota(axes.begin(), axes.end(), X.rank());
        return unsqueeze(std::forward<TensorT>(X), axes);
    }
    return X.view();
}

//==-------------------------------------------------------------------------
// Reorder operations used by DNN
//==-------------------------------------------------------------------------

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
inline reshape(const TensorT& src, TensorT& dst) {
    if (src.size() != dst.size())
        throw shape_error("cannot reshape to destination tensor");
    flat_copy(src, dst);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
inline broadcast(const TensorT& src, TensorT& dst) {
    reorder(src.broadcast(dst.shape()), dst);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
inline transpose(const TensorT& src, TensorT& dst, const std::vector<size_t>& perm) {
    reorder(src.transpose(perm), dst);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
inline slice(const TensorT& src, TensorT& dst,
             const std::vector<int>& starts, const std::vector<int>& ends,
             const std::vector<int>& axes, const std::vector<int>& steps)
{
    reorder(src.slice(starts, ends, axes, steps), dst);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
inline slice(const TensorT& src, TensorT& dst, const std::vector<SliceDim>& dims) {
    reorder(src.slice(dims), dst);
}

//==-------------------------------------------------------------------------
// Advanced reorder operations
//==-------------------------------------------------------------------------

/**
 * as_strided creates a view into the tensor given the exact strides and shape.
 * This means it manipulates the internal data structure of tensor and, if done
 * incorrectly, the tensor elements can point to invalid memory and can corrupt
 * results or crash your program. It is advisable to always use the original
 * strides when calculating new strides to avoid reliance on a contiguous
 * memory layout.
 *
 * Furthermore, tensors created with this function often contain self overlapping
 * memory, so that two elements are identical. Vectorized write operations on
 * such tensor will typically be unpredictable. They may even given different
 * results for small, large, or transposed tensors.
 *
 * For these reasons it is advisable to avoid as_strided when possible.
 *
 * @param X the tensor to create a view
 * @param shape the shape of the new tensor
 * @param strides the strides of the new tensor
 * @return the tensor view
 */
template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline as_strided(
    const TensorT& X,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides)
{
    return X.view(Shape::as_strided(shape, strides));
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
partition(const TensorT& X,
          std::vector<int> axes,
          std::vector<size_t> extents,
          std::vector<size_t> strides = {},
          std::vector<size_t> steps = {})
{
    if (axes.empty()) {
        axes.resize(X.rank());
        std::iota(axes.begin(), axes.end(), 0);
    } else {
        detail::norm_axes(X.rank(), axes);
    }

    assert(extents.size() == axes.size());
    assert(strides.empty() || strides.size() == axes.size());
    assert(steps.empty() || steps.size() == axes.size());

    if (strides.empty()) {
        strides.resize(axes.size());
        std::copy(extents.begin(), extents.end(), strides.begin());
    }
    if (steps.empty()) {
        steps.resize(axes.size());
        std::fill(steps.begin(), steps.end(), 1);
    }

    // Create strided shape
    auto rank = X.rank();
    std::vector<size_t> shape_extents, shape_strides;

    for (int k = 0; k < rank; k++) {
        auto it = std::find(axes.begin(), axes.end(), k);
        if (it == axes.end()) {
            shape_extents.push_back(X.extent(k));
            shape_strides.push_back(X.stride(k));
        } else {
            auto i = it - axes.begin();
            assert(extents[i] > 0 && strides[i] > 0 && steps[i] > 0);
            auto d = steps[i] * (extents[i] - 1) + 1;
            assert(X.extent(k) >= d);
            d = (X.extent(k) - d) / strides[i] + 1;
            shape_extents.push_back(d);
            shape_strides.push_back(d == 1 ? 0 : X.stride(k) * strides[i]);
        }
    }

    for (int k = 0; k < rank; k++) {
        auto it = std::find(axes.begin(), axes.end(), k);
        if (it == axes.end()) {
            shape_extents.push_back(1);
            shape_strides.push_back(0);
        } else {
            auto i = it - axes.begin();
            shape_extents.push_back(extents[i]);
            shape_strides.push_back(X.stride(k) * steps[i]);
        }
    }

    auto strided_shape = X.shape().as_strided(shape_extents, shape_strides);

    // Squeeze axes that not partitioned
    std::vector<int> sq;
    for (int k = 0; k < rank; k++) {
        if (std::find(axes.begin(), axes.end(), k) == axes.end())
            sq.push_back(rank + k);
    }
    if (!sq.empty()) {
        strided_shape = strided_shape.squeeze(sq);
    }

    // Return the partitioned view
    return X.view(strided_shape);
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline partition(const TensorT& X, int k, size_t n, size_t d, size_t s = 1) {
    return partition(X, std::vector<int>{k}, {n}, {d}, {s});
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline partition(const TensorT& X, int k, size_t n) {
    return partition(X, k, n, n, 1);
}

//==-------------------------------------------------------------------------
// Convenient routines
//==-------------------------------------------------------------------------

/**
 * Move axes of a tensor to new positions. Other axes remain in their original
 * order.
 *
 * @param X The tensor whose axes should be reordered.
 * @param source Original positions of the axes to move. These must be unique.
 * @param destination  Destination positions for each of the original axes.
 *        These must also be unique.
 * @return Tensor with moved axes. This tensor is a view of the input tensor.
 */
template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
moveaxis(const TensorT& X, std::vector<int> source, std::vector<int> destination) {
    detail::norm_axes(X.rank(), source);
    detail::norm_axes(X.rank(), destination);
    if (source.size() != destination.size())
        throw shape_error("moveaxis: source and destination axes must have the same number of elements");

    std::vector<size_t> order;
    for (int k = 0; k < X.rank(); ++k) {
        if (std::find(source.begin(), source.end(), k) == source.end())
            order.push_back(k);
    }
    for (int k = 0; k < X.rank(); ++k) {
        auto it = std::find(destination.begin(), destination.end(), k);
        if (it != destination.end())
            order.insert(order.begin()+k, source[it - destination.begin()]);
    }
    return X.transpose(order);
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
moveaxis(const TensorT& X, const int source, const int destination) {
    return moveaxis(X, std::vector<int>{source}, std::vector<int>{destination});
}

/**
 * Interchange two axes of a tensor.
 *
 * @param X Input tensor.
 * @param axis1 First axis
 * @param axis2 Second axis
 */
template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
swapaxes(const TensorT& X, int axis1, int axis2) {
    detail::norm_axes(X.rank(), axis1, axis2);
    std::vector<size_t> order(X.rank());
    std::iota(order.begin(), order.end(), 0);
    order[axis1] = axis2;
    order[axis2] = axis1;
    return X.transpose(order);
}

/**
 * Reverse the order of elements in a tensor along the given axis.
 *
 * The shape of the array is preserved, but the elements are reordered.
 *
 * @param X Input tensor.
 * @param axes Axes along which to flip over. The default, empty axes, will flip
 *        over all of the axes of the input tensor.
 */
template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
flip(const TensorT& X, std::vector<int> axes = {}) {
    // normalize axes
    if (axes.empty()) {
        axes.resize(X.rank());
        std::iota(axes.begin(), axes.end(), 0);
    } else {
        detail::norm_axes(X.rank(), axes, true);
    }

    std::vector<SliceDim> range;
    for (int k = 0; k < X.rank(); ++k) {
        if (std::find(axes.begin(), axes.end(), k) != axes.end()) {
            range.push_back({-1, std::numeric_limits<int>::lowest(), -1});
        } else {
            range.push_back({});
        }
    }
    return X.slice(range);
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline flip(const TensorT& X, const int axis) {
    return flip(X, std::vector<int>{axis});
}

/**
 * Rotate a tensor by 90 degrees in the plane specified by axes.
 *
 * Rotation direction is from the first towards the second axis.
 *
 * @param X Tensor of two or more dimensions.
 * @param axis1, axis2
 *        The tensor is rotated in the plane defined by the axes.
 *        Axes must be different.
 * @param k Number of times the tensor is rotated by 90 degrees.
 */
template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
rot90(const TensorT& X, int k = 1, int axis1 = -2, int axis2 = -1) {
    detail::norm_axes(X.rank(), axis1, axis2);
    switch (k % 4) {
    default: // k == 0
        return X.view();
    case 1:
        return swapaxes(flip(X, axis2), axis1, axis2);
    case 2:
        return flip(flip(X, axis1), axis2);
    case 3:
        return flip(swapaxes(X, axis1, axis2), axis2);
    }
}

//==-------------------------------------------------------------------------
// Concat and split
//==-------------------------------------------------------------------------

namespace concat_detail {
inline void check_input_shapes(int, std::vector<size_t>&) {}

template <typename TensorT, typename... Tensors>
void check_input_shapes(int axis, std::vector<size_t>& dims, const TensorT& input, const Tensors&... rest) {
    if (input.rank() != dims.size())
        throw shape_error("concat: all tensors to concat must have same rank");
    for (int i = 0; i < dims.size(); i++) {
        if (i == axis) {
            dims[i] += input.extent(i);
        } else if (input.extent(i) != dims[i]) {
            throw shape_error("concat: incompatible input tensor shape");
        }
    }
    check_input_shapes(axis, dims, rest...);
}

template <typename TensorT>
inline void do_concat(int, int, TensorT&) {}

template <typename TensorR, typename TensorT, typename... Tensors>
void do_concat(int axis, int offset, TensorR& output, const TensorT& input, const Tensors&... rest) {
    int next = offset + input.extent(axis);
    reorder(input, output.slice({offset}, {next}, {axis}, {1}));
    do_concat(axis, next, output, rest...);
}
} // namespace concat_detail

template <typename TensorT>
enable_if_tensor<TensorT, void>
concat(int axis, const std::vector<const TensorT*>& inputs, tensor_type<TensorT>& output) {
    if (inputs.empty())
        throw std::logic_error("concat: no input tensors");

    auto dims = inputs[0]->shape().extents();
    detail::norm_axis(dims.size(), axis);

    dims[axis] = 0;
    for (auto t : inputs)
        concat_detail::check_input_shapes(axis, dims, *t);
    output.resize(Shape(dims));

    int offset = 0;
    for (auto t : inputs) {
        int next = offset + t->extent(axis);
        reorder(*t, output.slice({offset}, {next}, {axis}, {1}));
        offset = next;
    }
}

template <typename TensorT, typename... Tensors>
std::enable_if_t<
    is_tensor<TensorT>::value &&
    cxx::conjunction<is_exactly_same_tensor<TensorT, Tensors>...>::value,
    tensor_type<TensorT>
>
concat(int axis, const TensorT& first, const Tensors&... rest) {
    detail::norm_axis(first.rank(), axis);

    auto dims = first.shape().extents();
    concat_detail::check_input_shapes(axis, dims, rest...);

    tensor_type<TensorT> output{Shape(dims)};
    concat_detail::do_concat(axis, 0, output, first, rest...);
    return output;
}

template <typename TensorT>
enable_if_tensor<TensorT, void>
split(int axis, const TensorT& input, const std::vector<tensor_type<TensorT>*>& outputs) {
    auto rank = input.rank();
    detail::norm_axis(rank, axis);

    size_t dim_sum = 0;
    for (auto t : outputs) {
        if (t->rank() != rank)
            throw shape_error("split: all output tensors must have same rank");
        for (int i = 0; i < rank; i++) {
            if (i == axis) {
                dim_sum += t->extent(i);
            } else if (t->extent(i) != input.extent(i)) {
                throw shape_error("split: incompatible output tensor shape");
            }
        }
    }
    if (dim_sum != input.extent(axis)) {
        throw shape_error("split: incompatible output tensor shape");
    }

    int offset = 0;
    for (auto t : outputs) {
        int next = offset + t->extent(axis);
        reorder(input.slice({offset}, {next}, {axis}, {1}), *t);
        offset = next;
    }
}

template <typename TensorT>
enable_if_tensor<TensorT, std::vector<tensor_view_type<TensorT>>>
split(const TensorT& input, int axis, const std::vector<size_t>& splits) {
    detail::norm_axis(input.rank(), axis);
    if (std::accumulate(splits.begin(), splits.end(), 0, std::plus<>()) != input.extent(axis))
        throw shape_error("split: invalid splits");

    std::vector<tensor_view_type<TensorT>> res;
    int offset = 0;
    for (int i = 0; i < splits.size(); i++) {
        int next = offset + splits[i];
        res.push_back(input.slice({offset}, {next}, {axis}, {1}));
        offset = next;
    }
    return res;
}

template <typename TensorT>
enable_if_tensor<TensorT, std::vector<tensor_view_type<TensorT>>>
split(const TensorT& input, int axis, int n_split) {
    assert(n_split > 0);
    detail::norm_axis(input.rank(), axis);

    int split_dim = input.extent(axis);
    int chunk_size = split_dim / n_split;
    int left_over = split_dim - chunk_size * n_split;
    std::vector<size_t> splits;
    for (int i = 0; i < n_split; i++)
        splits.push_back(i < left_over ? chunk_size+1 : chunk_size);
    return split(input, axis, splits);
}

} // namespace dlf
