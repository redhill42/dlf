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
    if (src_shape.stride(0) != 1 || static_cast<int>(src_shape.stride(1)) < src_shape.extent(0))
        return false;
    if (dst_shape.stride(1) != 1 || static_cast<int>(dst_shape.stride(0)) < dst_shape.extent(1))
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
inline slice(const TensorT& src, TensorT& dst, const std::vector<Range>& range) {
    reorder(src.slice(range), dst);
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
    const std::vector<size_t>& strides,
    const size_t offset = 0)
{
    return X.view(Shape::as_strided(shape, strides, offset));
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
partition(const TensorT& X,
          const std::vector<size_t> extents,
          const std::vector<size_t> strides = {},
          const std::vector<size_t> steps = {})
{
    assert(extents.size() <= X.rank());
    assert(strides.empty() || strides.size() == extents.size());
    assert(steps.empty() || steps.size() == extents.size());

    // Create strided shape
    auto rank = X.rank();
    auto skip = rank - extents.size();
    std::vector<size_t> shape_extents, shape_strides;

    for (int k = 0; k < skip; k++) {
        shape_extents.push_back(X.extent(k));
        shape_strides.push_back(X.stride(k));
    }

    for (int k = skip; k < rank; k++) {
        auto i = k - skip;
        auto extent = extents[i];
        auto stride = strides.empty() ? extent : strides[i];
        auto step   = steps.empty() ? 1 : steps[i];
        assert(extent > 0 && stride > 0 && step > 0);

        auto d = step * (extent - 1) + 1;
        assert(X.extent(k) >= d);
        d = (X.extent(k) - d) / stride + 1;

        shape_extents.push_back(d);
        shape_strides.push_back(d == 1 ? 0 : X.stride(k) * stride);
    }

    for (int k = skip; k < rank; k++) {
        auto i = k - skip;
        auto step = steps.empty() ? 1 : steps[i];
        shape_extents.push_back(extents[i]);
        shape_strides.push_back(X.stride(k) * step);
    }

    return as_strided(X, shape_extents, shape_strides, X.shape().offset());
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
partition(const TensorT& X, int axis, size_t extent, size_t stride, size_t step = 1) {
    assert(extent > 0 && stride > 0 && step > 0);
    detail::norm_axis(X.rank(), axis);
    auto shape_extents = X.shape().extents();
    auto shape_strides = X.shape().strides();

    auto d = step * (extent - 1) + 1;
    assert(X.extent(axis) >= d);
    d = (X.extent(axis) - d) / stride + 1;

    shape_extents[axis] = d;
    shape_strides[axis] = d == 1 ? 0 : X.stride(axis) * stride;
    shape_extents.insert(shape_extents.begin()+axis+1, extent);
    shape_strides.insert(shape_strides.begin()+axis+1, X.stride(axis) * step);

    return as_strided(X, shape_extents, shape_strides, X.shape().offset());
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
inline partition(const TensorT& X, int k, size_t n) {
    return partition(X, k, n, n, 1);
}

/**
 * Assemble a tensor from nested blocks.
 *
 * Blocks in the innermost tensors are concatenated along the last dimension,
 * then these are concatenated along the second-last dimension, and so on
 * until the outermost dimension is reached.
 *
 * Blocks can be of any dimension, but will not be broadcasted using the normal
 * rules. Instead, leading axes of size 1 are inserted, to make block rank the
 * same for all blocks. This is primarily useful for working with scalars.
 *
 * When the nested tensor is two levels deep, this allow block matrices to be
 * constructed from their components.
 */
template <typename TensorT>
enable_if_tensor<TensorT, void>
join(const Tensor<TensorT>& input, tensor_type<TensorT>& output) {
    // Determine the final rank
    auto rank = input.rank();
    for (const auto& b : input) {
        if (b.rank() > rank)
            rank = b.rank();
    }

    // Normalize to the same rank
    Tensor<tensor_view_type<TensorT>> blocks;
    transformTo(unsqueeze_left(input, rank), blocks, [rank](const auto& b){
        return unsqueeze_left(b, rank);
    });

    // Get dimensions on all axes
    std::vector<std::vector<size_t>> axes_dims(rank);
    std::vector<size_t> index(rank);

    for (int axis = 0; axis < rank; ++axis) {
        auto dim = blocks.extent(axis);
        axes_dims[axis].resize(dim);
        std::fill(index.begin(), index.end(), 0);
        for (int i = 0; i < dim; ++i) {
            index[axis] = i;
            const auto& b = blocks.data()[blocks.shape().offset(index)];
            axes_dims[axis][i] = b.extent(axis);
        }
    }

    // Check dimensions on all inner tensors
    std::fill(index.begin(), index.end(), 0);
    for (const auto& b : blocks) {
        for (int axis = 0; axis < rank; ++axis)
            if (b.extent(axis) != axes_dims[axis][index[axis]])
                throw shape_error("block: incompatible shape");
        blocks.shape().next(index);
    }

    // Calculate offsets into sliced final result
    std::vector<std::vector<int>> block_offsets(rank);
    for (int axis = 0; axis < rank; ++axis) {
        block_offsets[axis].resize(axes_dims[axis].size());
        std::partial_sum(axes_dims[axis].begin(), axes_dims[axis].end()-1,
                         block_offsets[axis].begin()+1); // first offset is 0
    }

    // Calculate the final shape
    std::vector<size_t> final_dims(rank);
    for (int i = 0; i < rank; ++i)
        final_dims[i] = std::accumulate(axes_dims[i].begin(), axes_dims[i].end(), 0);
    output.resize(Shape(final_dims));

    // Concatenate blocks into final result
    std::vector<Range> slice_range(rank);
    std::fill(index.begin(), index.end(), 0);
    for (const auto& b : blocks) {
        for (int i = 0; i < rank; ++i) {
            slice_range[i].start = block_offsets[i][index[i]];
            slice_range[i].end   = slice_range[i].start + b.extent(i);
        }
        reorder(b, output.slice(slice_range));
        blocks.shape().next(index);
    }
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT>>
join(const Tensor<TensorT>& input) {
    tensor_type<TensorT> output{};
    join(input, output);
    return output;
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

    std::vector<Range> range;
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

namespace detail {
template <typename TensorT>
void check_concat_shapes(int axis, std::vector<size_t>& dims, const TensorT& input) {
    if (input.rank() != dims.size())
        throw shape_error("concat: all tensors to concat must have same rank");
    for (int i = 0; i < dims.size(); i++) {
        if (i == axis) {
            dims[i] += input.extent(i);
        } else if (input.extent(i) != dims[i]) {
            throw shape_error("concat: incompatible input tensor shape");
        }
    }
}

template <typename TensorT, typename... Tensors>
inline void check_concat_shapes(int axis, std::vector<size_t>& dims, const TensorT& first, const Tensors&... rest) {
    check_concat_shapes(axis, dims, first);
    check_concat_shapes(axis, dims, rest...);
}

template <typename TensorR, typename TensorT>
void do_concat(int axis, int& offset, TensorR& output, const TensorT& input) {
    int next = offset + input.extent(axis);
    reorder(input, output.slice({offset}, {next}, {axis}, {1}));
    offset = next;
}

template <typename TensorR, typename TensorT, typename... Tensors>
inline void do_concat(int axis, int& offset, TensorR& output, const TensorT& first, const Tensors&... rest) {
    do_concat(axis, offset, output, first);
    do_concat(axis, offset, output, rest...);
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
        detail::check_concat_shapes(axis, dims, *t);
    output.resize(Shape(dims));

    int offset = 0;
    for (auto t : inputs) {
        detail::do_concat(axis, offset, output, *t);
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
    dims[axis] = 0;
    detail::check_concat_shapes(axis, dims, first, rest...);

    auto output = tensor_type<TensorT>{Shape(dims)};
    auto offset = 0;
    detail::do_concat(axis, offset, output, first, rest...);
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

//==-------------------------------------------------------------------------
// Gather and scatter
//==-------------------------------------------------------------------------

namespace detail {
inline int normalize_index(int index, const int max_item) {
    if (index < 0)
        index += max_item;
    if (index < 0)
        index = 0;
    if (index >= max_item)
        index = max_item-1;
    return index;
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_cpu_tensor<TensorX>::value>
gather(const TensorX& X, TensorY& Y, const TensorI& indices,
       int m, int n, int chunk, int max_item)
{
    tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 32, 0, n, 32), [&](auto r) {
        auto px = X.begin() + r.rows().begin() * chunk * max_item;
        for (int i = r.rows().begin(); i < r.rows().end(); ++i, px += chunk*max_item) {
            auto pi = indices.begin() + r.cols().begin();
            auto py = Y.begin() + (i*n + r.cols().begin()) * chunk;
            for (int j = r.cols().size(); j > 0; --j, ++pi, py += chunk) {
                auto id = normalize_index(*pi, max_item);
                std::copy(px + id*chunk, px + (id+1)*chunk, py);
            }
        }
    });
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_gpu_tensor<TensorX>::value>
gather(const TensorX& X, TensorY& Y, const TensorI& indices,
       int m, int n, int chunk, int max_item)
{
    gpgpu::dnn::gather(
        m, n, chunk, max_item,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        Y.shape().extents(), Y.shape().strides(),
        Y.data(), Y.shape().offset());
}
} // namespace detail

/**
 * Take elements from an array along an axis.
 */
template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorI, tensor_type<TensorX, int>>::value &&
    !std::is_const<std::remove_reference_t<TensorY>>::value>
gather(const TensorX& X, TensorY&& Y, const TensorI& indices, int axis = 0) {
    auto r = static_cast<int>(X.rank());
    auto q = static_cast<int>(indices.rank());

    if (r == 0)
        throw shape_error("gather: input tensor must have rank >= 1");
    detail::norm_axis(r, axis);

    int nd = q + r - 1;
    int m, n, chunk;
    std::vector<size_t> dims;

    m = n = chunk = 1;
    for (int i = 0; i < nd; i++) {
        size_t dim;
        if (i < axis) {
            dim = X.extent(i);
            m *= dim;
        } else if (i < axis + q) {
            dim = indices.extent(i - axis);
            n *= dim;
        } else {
            dim = X.extent(i - q + 1);
            chunk *= dim;
        }
        dims.push_back(dim);
    }

    Y.resize(Shape(dims));
    detail::gather(X, Y, indices, m, n, chunk, X.extent(axis));
}

template <typename TensorT, typename TensorI>
std::enable_if_t<
    is_exactly_same_tensor<TensorI, tensor_type<TensorT, int>>::value,
    tensor_type<TensorT>>
gather(const TensorT& X, const TensorI& indices, int axis = 0) {
    tensor_type<TensorT> Y{};
    gather(X, Y, indices, axis);
    return Y;
}

namespace detail {
template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_cpu_tensor<TensorX>::value, void>
gather_elements(const TensorX& X, TensorY& Y, const TensorI& indices, int axis) {
    tbb::parallel_for(tbb::blocked_range<int>(0, indices.size(), GRAINSIZE), [&](auto r) {
        const auto i_stride1 = indices.shape().partial_size(axis+1, indices.rank());
        const auto i_stride2 = i_stride1 * indices.extent(axis);
        const auto x_stride1 = X.shape().partial_size(axis+1, X.rank());
        const auto x_stride2 = x_stride1 * X.extent(axis);

        auto px = X.data();
        auto pi = indices.begin() + r.begin();
        auto py = Y.begin() + r.begin();

        const bool x_contiguous = X.shape().is_contiguous();
        const auto x_offset = X.shape().offset();
        const auto max_item = static_cast<int>(X.extent(axis));

        for (int id = r.begin(); id < r.end(); ++id, ++pi, ++py) {
            auto tmp = normalize_index(*pi, max_item);
            auto x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
            *py = px[x_contiguous ? x_id + x_offset : X.shape().linear_offset(x_id)];
        }
    });
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_gpu_tensor<TensorX>::value, void>
gather_elements(const TensorX& X, TensorY& Y, const TensorI& indices, int axis) {
    gpgpu::dnn::gather_elements(
        Y.size(), axis,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        Y.shape().extents(), Y.shape().strides(),
        Y.data(), Y.shape().offset());
}
} // namespace detail

/**
 * Takes two inputs data and indices of the same rank r >= 1 and an optional
 * attribute axis that identifies an axis of data (by default, the outer-most
 * axis, that is axis 0). It is an indexing operation that produces its output
 * by indexing into the input data tensor at index positions determined by
 * elements of the indices tensor. Its output shape is same as the shape of
 * indices and consists of one value (gathered from the data) for each element
 * in indices.
 *
 * For instance, in the 3-D case (r = 3), the output produced is determined by
 * the following equations:
 *
 *   out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0
 *   out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1
 *   out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2
 */
template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorI, tensor_type<TensorX, int>>::value &&
    !std::is_const<std::remove_reference_t<TensorY>>::value>
gather_elements(const TensorX& X, TensorY&& Y, const TensorI& indices, int axis = 0) {
    if (X.rank() != indices.rank())
        throw shape_error("gather_elements: shape mismatch");
    for (int i = 0; i < X.rank(); i++)
        if (indices.extent(i) > X.extent(i))
            throw shape_error("gather_elements: shape mismatch");

    detail::norm_axis(X.rank(), axis);
    Y.resize(indices.shape());
    detail::gather_elements(X, Y, indices, axis);
}

template <typename TensorX, typename TensorI>
std::enable_if_t<
    is_exactly_same_tensor<TensorI, tensor_type<TensorX, int>>::value,
    tensor_type<TensorX>>
gather_elements(const TensorX& X, const TensorI& indices, int axis = 0) {
    tensor_type<TensorX> Y{};
    gather_elements(X, Y, indices, axis);
    return Y;
}

namespace detail {
template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<is_cpu_tensor<TensorX>::value>
scatter_elements(TensorX& X, const TensorI& indices, const TensorY& updates, int axis) {
    tbb::parallel_for(tbb::blocked_range<int>(0, updates.size(), GRAINSIZE), [&](auto r) {
        const auto i_stride1 = indices.shape().partial_size(axis+1, indices.rank());
        const auto i_stride2 = i_stride1 * indices.extent(axis);
        const auto x_stride1 = X.shape().partial_size(axis+1, X.rank());
        const auto x_stride2 = x_stride1 * X.extent(axis);

        auto px = X.data();
        auto pi = indices.begin() + r.begin();
        auto pu = updates.begin() + r.begin();

        const bool x_contiguous = X.shape().is_contiguous();
        const auto x_offset = X.shape().offset();
        const auto max_item = static_cast<int>(X.extent(axis));

        for (int id = r.begin(); id < r.end(); ++id, ++pu, ++pi) {
            auto tmp = normalize_index(*pi, max_item);
            auto x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
            px[x_contiguous ? x_id + x_offset : X.shape().linear_offset(x_id)] = *pu;
        }
    });
}

template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<is_gpu_tensor<TensorX>::value>
scatter_elements(TensorX& X, const TensorI& indices, const TensorY& updates, int axis) {
    gpgpu::dnn::scatter_elements(
        indices.size(), axis,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        updates.shape().extents(), updates.shape().strides(),
        updates.data(), updates.shape().offset());
}
} // namespace detail

/**
 * Takes three inputs data, updates, and indices of the same rank r >= 1 and
 * an optional axis that identifies an axis of data (by default, the outer-most
 * axis, that is axis 0). The operation updates data to values specified by
 * updates at specific index positions specified by indices.
 *
 * For each entry in updates, the target index in data is obtained by combining
 * the corresponding entry in indices with the index of the entry itself:
 * the index-value for dimension = axis is obtained from the value of the
 * corresponding entry in indices and the index value for dimension != axis is
 * obtained from the index of the entry itself.
 *
 * For instance, in a 2-D tensor case, the update corresponding to the [i][j]
 * entry is performed as below:
 *
 *    output[indices[i][j]][j] = updates[i][j] if axis = 0
 *    output[i][indices[i][j]] = updates[i][j] if axis = 1
 */
template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorI, tensor_type<TensorX, int>>::value>
scatter_elements(TensorX& X, const TensorI& indices, const TensorY& updates, int axis = 0) {
    if (X.rank() != updates.rank() || X.rank() != indices.rank())
        throw shape_error("scatter_elements: shape mismatch");
    if (updates.shape() != indices.shape())
        throw shape_error("scatter_elements: shape mismatch");

    detail::norm_axis(X.rank(), axis);
    detail::scatter_elements(X, indices, updates, axis);
}

namespace detail {
template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_cpu_tensor<TensorX>::value>
gather_nd(const TensorX& X, TensorY& Y, const TensorI& indices,
          const int n, const int k, const int chunk)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 64), [&](auto r) {
        auto px = X.begin();
        auto py = Y.begin() + r.begin()*chunk;
        auto pi = indices.begin() + r.begin()*k;
        auto dims = X.shape().extents();

        for (int i = r.begin(); i < r.end(); ++i) {
            // compute slice offset
            int offset = 0, dim = 1;
            for (int j = 0; j < k; ++j, ++pi) {
                offset = offset*dim + normalize_index(*pi, dims[j]);
                dim = dims[j];
            }
            offset *= chunk;

            // copy slice
            std::copy(px+offset, px+offset+chunk, py);
            py += chunk;
        }
    });
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_gpu_tensor<TensorX>::value>
gather_nd(const TensorX& X, TensorY& Y, const TensorI& indices,
          const int n, const int k, const int chunk)
{
    gpgpu::dnn::gather_nd(
        n, k, chunk,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        Y.shape().extents(), Y.shape().strides(),
        Y.data(), Y.shape().offset());
}
} // namespace detail

/**
 * Given data tensor of rank r >= 1, and indices tensor of rank q >= 1, this
 * routine gathers slices of data into an output tensor of rank q + r -
 * indices_shape[-1] - 1.
 *
 * indices is a q-dimensional integer tensor, best thought of as a (q-1)-dimensional
 * tensor of index-tuples into data, where each element defines a slice of data.
 *
 * Some salient points about the inputs' rank and shape:
 *
 *   1. r >= 1 and q >= 1 are to be honored. There is no dependency condition
 *      to be met between ranks r and q.
 *
 *   2. The indices_shape[-1] should have a value between 1 (inclusive) and
 *      rank r (inclusive).
 *
 *   3. All values in indices are expected to be within bounds [-s, s-1] along
 *      axis of size s (i.e.) -data_shape[i] <= indices[i...,i] <= data_shape[i]-1.
 *      It is an error if any of the index values are out of bounds.
 *
 * The output is computed as follows:
 *
 * The output tensor is obtained by mapping each index-tuple in the indices tensor
 * to the corresponding slice of the input data.
 *
 *   1. If indices-shape[-1] > r => error condition.
 *
 *   2. If indices-shape[-1] == r, since the rank of indices is q, indices can be
 *      thought of as a (q-1)-dimensional tensor containing 1-D tensors of
 *      dimension r. Let us think of each such r ranked tensor as indices_slice.
 *      Each scalar value corresponding to data[indices_slice] is filled into
 *      the corresponding location of the (q-1)-dimensional tensor to form the
 *      output tensor.
 *
 *   3. If indices_shape[-1] < r, since the rank of indices is q, indices can be
 *      thought of as a (q-1)-dimensional tensor containing 1-D tensors of
 *      dimension < r. Let us think of each such tensors as indices_slice. Each
 *      tensor slice corresponding to data[indices_slice, :] is filled into the
 *      corresponding location of the (q-1)-dimensional tensor to form the output
 *      tensor.
 *
 */
template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorI, tensor_type<TensorX, int>>::value &&
    !std::is_const<std::remove_reference_t<TensorY>>::value>
gather_nd(const TensorX& X, TensorY&& Y, const TensorI& indices) {
    auto r = static_cast<int>(X.rank());
    auto q = static_cast<int>(indices.rank());

    if (r == 0)
        throw shape_error("gather_nd: input tensor must have rank >= 1");
    if (q == 0)
        throw shape_error("gather_nd: indices tensor must have rank >= 1");

    auto k = static_cast<int>(indices.extent(-1));
    if (k > r)
        throw shape_error("gather_nd: last dimension of indices tensor must no be larger than the rank of input tensor");

    std::vector<size_t> dims;
    int n = 1, chunk = 1;
    for (int i = 0; i < q-1; i++) {
        dims.push_back(indices.extent(i));
        n *= indices.extent(i);
    }
    for (int i = k; i < r; i++) {
        dims.push_back(X.extent(i));
        chunk *= X.extent(i);
    }

    Y.resize(Shape(dims));
    detail::gather_nd(X, Y, indices, n, k, chunk);
}

template <typename TensorX, typename TensorI>
std::enable_if_t<
    is_exactly_same_tensor<TensorI, tensor_type<TensorX, int>>::value,
    tensor_type<TensorX>>
gather_nd(const TensorX& X, const TensorI& indices) {
    tensor_type<TensorX> Y{};
    gather_nd(X, Y, indices);
    return Y;
}

namespace detail {
template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<is_cpu_tensor<TensorX>::value>
scatter_nd(TensorX& X, const TensorI& indices, const TensorY& updates,
           const int n, const int k, const int chunk)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 1), [&](auto r) {
        auto px = X.begin();
        auto py = updates.begin() + r.begin()*chunk;
        auto pi = indices.begin() + r.begin()*k;
        auto dims = X.shape().extents();

        for (int i = r.begin(); i < r.end(); ++i) {
            // compute slice offset
            int offset = 0, dim = 1;
            for (int j = 0; j < k; ++j, ++pi) {
                offset = offset*dim + detail::normalize_index(*pi, dims[j]);
                dim = dims[j];
            }
            offset *= chunk;

            // copy slice
            std::copy(py, py+chunk, px+offset);
            py += chunk;
        }
    });
}

template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<is_gpu_tensor<TensorX>::value>
scatter_nd(TensorX& X, const TensorI& indices, const TensorY& updates,
           const int n, const int k, const int chunk)
{
    gpgpu::dnn::scatter_nd(
        n, k, chunk,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        updates.shape().extents(), updates.shape().strides(),
        updates.data(), updates.shape().offset());
}
} // namespace detail

/**
 * Takes three inputs, data tensor of rank r >= 1, indices tensor of rank q >= 1,
 * and updates tensor of rank q + r - indices.shape[-1] - 1. Updating data inplace
 * to values specified by updates at specific index positions specified by indices.
 *
 * `indices` is an integer tensor. Let k denote indices.shape[-1], the last
 * dimension in the shape of indices. indices is treated as a (q-1)-dimensional
 * tensor of k-tuples, where each k-tuple is a partial-index into data. Hence,
 * k can be a value at most the rank of data. When k equals rank(data), each update
 * entry specifies an update to a single element of the tensor. When k is less than
 * rank(data) each update entry specifies an update to a slice of the tensor.
 *
 * `updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values.
 * Thus, the first (q-1) dimensions of updates.shape must match the first (q-1)
 * dimensions of indices.shape. The remaining dimensions of `updates` correspond
 * to the dimensions of the replacement-slice-values. Each replacement-slice-value
 * is a (r-k) dimensional tensor, corresponding to the trailing (r-k) dimensions
 * of data. Thus, the shape of `updates` must equal indices.shape[0:q-1]++data.shape[k:r-1],
 * where ++ denotes the concatenation of shapes.
 *
 */
template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorI, tensor_type<TensorX, int>>::value>
scatter_nd(TensorX& X, const TensorI& indices, const TensorY& updates) {
    auto r = static_cast<int>(X.rank());
    auto q = static_cast<int>(indices.rank());

    if (r == 0)
        throw shape_error("scatter_nd: input tensor must have rank >= 1");
    if (q == 0)
        throw shape_error("scatter_nd: indices tensor must have rank >= 1");

    auto k = static_cast<int>(indices.extent(-1));
    if (k > r)
        throw shape_error("scatter_nd: last dimension of indices tensor must not be larger than the rank of input tensor");

    std::vector<size_t> dims;
    int n = 1, chunk = 1;
    for (int i = 0; i < q-1; i++) {
        dims.push_back(indices.extent(i));
        n *= indices.extent(i);
    }
    for (int i = k; i < r; i++) {
        dims.push_back(X.extent(i));
        chunk *= X.extent(i);
    }
    if (updates.shape() != Shape(dims)) {
        throw shape_error("scatter_nd: updates tensor has incorrect dimensions");
    }

    detail::scatter_nd(X, indices, updates, n, k, chunk);
}

} // namespace dlf
