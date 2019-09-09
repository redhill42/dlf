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
// Pad
//==-------------------------------------------------------------------------

enum class PadMode {
    Constant, Edge, Reflect, Symmetric
};

namespace detail {
template <typename TensorT>
inline void prepend_constant(TensorT& X, const tensor_value_type<TensorT>& val, int pad_amt, int axis) {
    if (pad_amt > 0) {
        X.slice({0}, {pad_amt}, {axis}, {1}).fill(val);
    }
}

template <typename TensorT>
inline void append_constant(TensorT& X, const tensor_value_type<TensorT>& val, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto dim = static_cast<int>(X.extent(axis));
        X.slice({-pad_amt}, {dim}, {axis}, {1}).fill(val);
    }
}

template <typename TensorT>
void prepend_edge(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto x_edge_slice = X.slice({pad_amt}, {pad_amt+1}, {axis}, {1});
        auto y_edge_slice = X.slice({0}, {pad_amt}, {axis}, {1});
        reorder(x_edge_slice.broadcast(y_edge_slice.shape()), y_edge_slice);
    }
}

template <typename TensorT>
void append_edge(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto dim = static_cast<int>(X.extent(axis));
        auto x_edge_slice = X.slice({-pad_amt-1}, {-pad_amt}, {axis}, {1});
        auto y_edge_slice = X.slice({-pad_amt}, {dim}, {axis}, {1});
        reorder(x_edge_slice.broadcast(y_edge_slice.shape()), y_edge_slice);
    }
}

template <typename TensorT>
void prepend_reflect(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto x_edge_slice = X.slice({2*pad_amt}, {pad_amt}, {axis}, {-1});
        auto y_edge_slice = X.slice({0}, {pad_amt}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
}

template <typename TensorT>
void append_reflect(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto dim = static_cast<int>(X.extent(axis));
        auto x_edge_slice = X.slice({-pad_amt-2}, {-2*pad_amt-2}, {axis}, {-1});
        auto y_edge_slice = X.slice({-pad_amt}, {dim}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
}

template <typename TensorT>
void prepend_symmetric(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto x_edge_slice = X.slice({2*pad_amt-1}, {pad_amt-1}, {axis}, {-1});
        auto y_edge_slice = X.slice({0}, {pad_amt}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
}

template <typename TensorT>
void append_symmetric(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto dim = static_cast<int>(X.extent(axis));
        auto x_edge_slice = X.slice({-pad_amt-1}, {-2*pad_amt-1}, {axis}, {-1});
        auto y_edge_slice = X.slice({-pad_amt}, {dim}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
}
} // namespace detail

/**
 * Pad a tensor.
 *
 * @param X The input tensor.
 * @param Y The output tensor.
 * @param pads List of integers indicating the number of padding elements to
 *        add or remove (if negative) at the beginning and end of each axis.
 *        For 2D it is the number of pixels. `pads` rank should be double of
 *        the input's rank. `pads` format should be as follow [x1_begin,
 *        x2_begin,...,x1_end,x2_end,...], where xi_begin, the number of pixels
 *        added at the beginning of axis `i`, and xi_end, the number of pixels
 *        added at the end of axis `i`.
 * @param mode Pad mode. One of the following values:
 *        Constant:
 *            Pads with a constant value.
 *        Edge:
 *            Pads with the edge values.
 *        Reflect:
 *            Pads with the reflection of the vector mirrored on the first
 *            and last values of the vector along each axis.
 *        Symmetric:
 *            Pads with the reflection of the vector mirrored along the
 *            edge.
 * @param val When mode is 'constant', the value to be filled.
 */
template <typename TensorT>
enable_if_tensor<TensorT, void>
pad(const TensorT& X, tensor_type<TensorT>& Y,
    const std::vector<int>& pads, const PadMode mode = PadMode::Constant,
    const tensor_value_type<TensorT>& val = tensor_value_type<TensorT>{})
{
    // Validate pads and calculate output shape
    std::vector<size_t> y_dims;
    auto rank = X.rank();
    if (pads.size() != rank*2)
        throw std::invalid_argument("pad: the 'pads' argument has incorrect rank");
    for (int i = 0; i < rank; i++) {
        auto old_dim = static_cast<int>(X.extent(i));
        auto new_dim = old_dim + pads[i] + pads[i + rank];
        if (new_dim <= 0 || (pads[i]<0 && -pads[i]>old_dim) || (pads[i+rank]<0 && -pads[i+rank]>old_dim))
            throw shape_error("pad: the 'pads' argument contains invalid value");
        y_dims.push_back(new_dim);
    }
    Y.resize(Shape(y_dims));

    // Copy core data
    std::vector<Range> x_slice, y_slice;
    for (int i = 0; i < rank; i++) {
        int x_start = 0, x_end = X.extent(i);
        int y_start = 0, y_end = Y.extent(i);
        if (pads[i] < 0) {
            x_start = -pads[i];
        } else {
            y_start = pads[i];
        }
        if (pads[i+rank] < 0) {
            x_end += pads[i+rank];
        } else {
            y_end -= pads[i+rank];
        }
        x_slice.push_back({x_start, x_end});
        y_slice.push_back({y_start, y_end});
    }
    reorder(X.slice(x_slice), Y.slice(y_slice));

    // Padding
    switch (mode) {
    case PadMode::Constant:
        for (int axis = 0; axis < rank; axis++) {
            detail::prepend_constant(Y, val, pads[axis], axis);
            detail::append_constant(Y, val, pads[axis+rank], axis);
        }
        break;

    case PadMode::Edge:
        for (int axis = 0; axis < rank; axis++) {
            detail::prepend_edge(Y, pads[axis], axis);
            detail::append_edge(Y, pads[axis+rank], axis);
        }
        break;

    case PadMode::Reflect:
        for (int axis = 0; axis < rank; axis++) {
            detail::prepend_reflect(Y, pads[axis], axis);
            detail::append_reflect(Y, pads[axis+rank], axis);
        }
        break;

    case PadMode::Symmetric:
        for (int axis = 0; axis < rank; axis++) {
            detail::prepend_symmetric(Y, pads[axis], axis);
            detail::append_symmetric(Y, pads[axis+rank], axis);
        }
        break;

    default:
        throw std::logic_error("pad: unsupported mode");
    }
}

template <typename TensorT>
enable_if_tensor<TensorT>
pad(const TensorT& X, const std::vector<int>& pads,
    const PadMode mode = PadMode::Constant,
    const tensor_value_type<TensorT>& val = tensor_value_type<TensorT>{})
{
    tensor_type<TensorT> Y{};
    pad(X, Y, pads, mode, val);
    return Y;
}

} // namespace dlf
