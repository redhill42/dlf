#pragma once

#include "./reorder_detail.h"

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor reorder operations
//==-------------------------------------------------------------------------

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
template <typename Block, typename TensorR>
std::enable_if_t<
    is_cpu_tensor<Block>::value &&
    is_tensor<tensor_value_type<Block>>::value &&
    is_exactly_same_tensor<tensor_value_type<Block>, TensorR>::value &&
    !std::is_const<std::remove_reference<TensorR>>::value>
join(const Block& input, TensorR&& output) {
    // Determine the final rank
    auto rank = input.rank();
    for (const auto& b : input) {
        if (b.rank() > rank)
            rank = b.rank();
    }

    // Normalize to the same rank
    Tensor<tensor_view_type<tensor_value_type<Block>>> blocks{};
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

template <typename Block>
std::enable_if_t<
    is_cpu_tensor<Block>::value && is_tensor<tensor_value_type<Block>>::value,
    tensor_type<tensor_value_type<Block>>>
join(const Block& input) {
    tensor_type<tensor_value_type<Block>> output{};
    join(input, output);
    return output;
}

/**
 * Construct a tensor by repeating X the number of times given by reps.
 *
 * If reps has length d, the result will have dimension of max(d, X.rank).
 *
 * If X.rank < d, A is promoted to be d-dimensional by prepending new axes.
 * If X.rank > d, reps is promoted to X.rank by pre-pending 1's to it.
 *
 * @param X the input tensor.
 * @param Y the output tensor.
 * @param reps the number of repetitions of X along each axis.
 */
template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference<TensorR>>::value>
tile(const TensorT& X, TensorR&& Y, std::vector<size_t> reps) {
    auto rank = std::max(X.rank(), reps.size());
    auto x_view = unsqueeze_left(X, rank);
    reps.insert(reps.begin(), rank - reps.size(), 1);
    join(Tensor<tensor_view_type<TensorT>>(Shape(reps), x_view), std::forward<TensorR>(Y));
}

template <typename TensorT>
enable_if_tensor<TensorT>
tile(const TensorT& X, const std::vector<size_t>& reps) {
    tensor_type<TensorT> Y{};
    tile(X, Y, reps);
    return Y;
}

/**
 * Repeat elements of a tensor.
 */
template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference<TensorR>>::value>
repeat(const TensorT& X, TensorR&& Y, int repeats, int axis) {
    assert(repeats > 0);
    detail::norm_axis(X.rank(), axis);
    std::vector<size_t> reps(X.rank(), 1);
    reps[axis] = repeats;
    tile(X, std::forward<TensorR>(Y), reps);
}

template <typename TensorT>
enable_if_tensor<TensorT>
repeat(const TensorT& X, int repeats, int axis) {
    tensor_type<TensorT> Y{};
    repeat(X, Y, repeats, axis);
    return Y;
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

template <typename Iterator, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorR, typename std::iterator_traits<Iterator>::value_type>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
concat(int axis, Iterator first, Iterator last, TensorR&& output) {
    auto rank = first->rank();
    detail::norm_axis(rank, axis);

    std::vector<size_t> dims(rank, 1);
    dims[axis] = std::distance(first, last);

    using TensorT = tensor_view_type<typename std::iterator_traits<Iterator>::value_type>;
    auto block = Tensor<TensorT>(Shape(dims));
    std::transform(first, last, block.begin(), [](const auto& x){ return x.view(); });
    join(block, output);
}

template <typename Iterator>
std::enable_if_t<
    is_tensor<typename std::iterator_traits<Iterator>::value_type>::value,
    tensor_type<typename std::iterator_traits<Iterator>::value_type>>
concat(int axis, Iterator first, Iterator last) {
    tensor_type<typename std::iterator_traits<Iterator>::value_type> Y{};
    concat(axis, first, last, Y);
    return Y;
}

template <typename TensorT, typename... Tensors>
std::enable_if_t<
    is_tensor<TensorT>::value &&
    cxx::conjunction<is_exactly_same_tensor<TensorT, Tensors>...>::value,
    tensor_type<TensorT>>
concat(int axis, const TensorT& first, const Tensors&... rest) {
    auto rank = first.rank();
    detail::norm_axis(rank, axis);

    std::vector<size_t> dims(rank, 1);
    dims[axis] = 1 + sizeof...(rest);
    return join(Tensor<tensor_view_type<TensorT>>(Shape(dims), {first.view(), rest.view()...}));
}

template <typename TensorT, typename Iterator>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, typename std::iterator_traits<Iterator>::value_type>::value>
split(const TensorT& input, int axis, Iterator first, Iterator last) {
    auto rank = input.rank();
    detail::norm_axis(rank, axis);

    size_t dim_sum = 0;
    for (auto it = first; it != last; ++it) {
        if (it->rank() != rank)
            throw shape_error("split: all output tensors must have same rank");
        for (int i = 0; i < rank; ++i) {
            if (i == axis) {
                dim_sum += it->extent(i);
            } else if (it->extent(i) != input.extent(i)) {
                throw shape_error("split: incompatible output tensor shape");
            }
        }
    }
    if (dim_sum != input.extent(axis)) {
        throw shape_error("split: incompatible output tensor shape");
    }

    int offset = 0;
    for (auto it = first; it != last; ++it) {
        int next = offset + it->extent(axis);
        reorder(input.slice({offset}, {next}, {axis}, {1}), *it);
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

template <typename TensorT>
enable_if_tensor<TensorT>
erase(const TensorT& X, int axis, const std::vector<Range>& range) {
    detail::norm_axis(X.rank(), axis);
    auto max_item = X.extent(axis);

    std::unordered_set<int> erased_items;
    for (auto r : range) {
        r = r.normalize(max_item);
        for (int i = r.start, n = r.size(); n > 0; --n, i += r.step)
            erased_items.insert(i);
    }
    if (erased_items.size() == max_item) {
        throw std::logic_error("erase: cannot erase all items in an axis");
    }

    std::vector<tensor_view_type<TensorT>> slices;
    for (int start = 0; start < max_item; start++) {
        if (erased_items.find(start) == erased_items.end()) {
            int end = start+1;
            for (; end < max_item; end++) {
                if (erased_items.find(end) != erased_items.end())
                    break;
            }
            slices.push_back(X.slice({start}, {end}, {axis}, {1}));
            start = end;
        }
    }

    return concat(axis, slices.begin(), slices.end());
}

template <typename TensorT, typename TensorU>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorU>::value,
    tensor_type<TensorT>>
insert(const TensorT& X, int axis, const std::vector<int>& insertions, const TensorU& values) {
    detail::norm_axis(X.rank(), axis);
    auto max_item = X.extent(axis);

    tensor_view_type<TensorT> value_view = values.view();
    if (value_view.is_scalar()) {
        auto dims = X.shape().extents();
        dims[axis] = 1;
        value_view = value_view.broadcast(Shape(dims));
    }

    std::vector<tensor_view_type<TensorT>> slices;
    for (int start = 0; start < max_item; ) {
        if (std::find(insertions.begin(), insertions.end(), start) != insertions.end()) {
            slices.push_back(value_view);
        }

        int end = start+1;
        for (; end < max_item; ++end) {
            if (std::find(insertions.begin(), insertions.end(), end) != insertions.end())
                break;
        }
        slices.push_back(X.slice({start}, {end}, {axis}, {1}));
        start = end;
    }
    if (std::find(insertions.begin(), insertions.end(), max_item) != insertions.end()) {
        slices.push_back(value_view);
    }

    return concat(axis, slices.begin(), slices.end());
}

template <typename TensorT, typename TensorU>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorU>::value,
    tensor_type<TensorT>>
replace(const TensorT& X, int axis, const std::vector<Range>& range, const TensorU& values) {
    detail::norm_axis(X.rank(), axis);
    auto max_item = X.extent(axis);

    tensor_view_type<TensorT> value_view = values.view();
    if (value_view.is_scalar()) {
        auto dims = X.shape().extents();
        dims[axis] = 1;
        value_view = value_view.broadcast(Shape(dims));
    }

    std::unordered_set<int> erased_items;
    for (auto r : range) {
        r = r.normalize(max_item);
        for (int i = r.start, n = r.size(); n > 0; --n, i += r.step)
            erased_items.insert(i);
    }

    std::vector<tensor_view_type<TensorT>> slices;
    for (int start = 0; start < max_item; ) {
        if (erased_items.find(start) != erased_items.end()) {
            slices.push_back(value_view);
            for (++start; start < max_item; ++start) {
                if (erased_items.find(start) == erased_items.end())
                    break;
            }
        } else {
            int end = start + 1;
            for (; end < max_item; ++end) {
                if (erased_items.find(end) != erased_items.end())
                    break;
            }
            slices.push_back(X.slice({start}, {end}, {axis}, {1}));
            start = end;
        }
    }

    return concat(axis, slices.begin(), slices.end());
}

//==-------------------------------------------------------------------------
// Gather and scatter
//==-------------------------------------------------------------------------

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

//==-------------------------------------------------------------------------
// Merge and sort
//==-------------------------------------------------------------------------

/**
 * Merges two sorted tensors into one sorted tensor. The input and output
 * tensors are sorted on the given axis.
 */
template <typename LHS, typename RHS, typename RET, typename Compare>
std::enable_if_t<
    is_exactly_same_tensor<LHS, RET>::value &&
    is_exactly_same_tensor<RHS, RET>::value &&
    !std::is_const<std::remove_reference_t<RET>>::value>
merge(const LHS& A, const RHS& B, RET&& C, int axis, Compare comp) {
    auto rank = A.rank();
    if (B.rank() != rank)
        throw shape_error("merge: incompatible shape");

    std::vector<size_t> dims(rank);
    detail::norm_axis(rank, axis);
    for (int i = 0; i < rank; i++) {
        if (i == axis) {
            dims[i] = A.extent(i) + B.extent(i);
        } else if (A.extent(i) != B.extent(i)) {
            throw shape_error("merge: incompatible shape");
        } else {
            dims[i] = A.extent(i);
        }
    }
    C.resize(Shape(dims));

    auto a_view = moveaxis(A, axis, -1);
    auto b_view = moveaxis(B, axis, -1);
    auto c_view = moveaxis(C, axis, -1);
    detail::merge(a_view.shape(), a_view.data(),
                  b_view.shape(), b_view.data(),
                  c_view.shape(), c_view.data(),
                  comp);
}

template <typename LHS, typename RHS, typename RET>
std::enable_if_t<
    is_exactly_same_tensor<LHS, RET>::value &&
    is_exactly_same_tensor<RHS, RET>::value &&
    !std::is_const<std::remove_reference_t<RET>>::value>
inline merge(const LHS& A, const RHS& B, RET&& C, int axis = -1) {
    merge(A, B, C, axis, std::less<>());
}

template <typename LHS, typename RHS, typename Compare>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
inline merge(const LHS& A, const RHS& B, int axis, Compare comp) {
    tensor_type<LHS> C{};
    merge(A, B, C, axis, comp);
    return C;
}

template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
inline merge(const LHS& A, const RHS& B, int axis = -1) {
    return merge(A, B, axis, std::less<>());
}

template <typename TensorT, typename Compare>
std::enable_if_t<is_cpu_tensor<TensorT>::value>
sort(TensorT& X, int axis, Compare comp) {
    auto x_view = moveaxis(X, axis, -1);
    detail::sort(x_view.shape(), x_view.data(), comp);
}

template <typename TensorT>
std::enable_if_t<is_cpu_tensor<TensorT>::value>
inline sort(TensorT& X, int axis = -1) {
    sort(X, axis, std::less<>());
}

} // namespace dlf
