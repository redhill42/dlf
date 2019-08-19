#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor reorder operations
//==-------------------------------------------------------------------------

namespace detail {
template <typename T>
void reorder(const Shape& src_shape, const T* src_data, const size_t src_size,
             const Shape& dst_shape, T* dst_data)
{
    assert(src_shape == dst_shape);

    if (dst_shape.is_contiguous()) {
        if (src_size == 1) {
            std::fill(dst_data + dst_shape.offset(),
                      dst_data + dst_shape.offset() + dst_shape.size(),
                      src_data[src_shape.offset()]);
            return;
        }

        if (src_shape.is_contiguous()) {
            if (src_data != dst_data || src_shape.offset() != dst_shape.offset()) {
                par::copy(src_data + src_shape.offset(),
                          src_data + src_shape.offset() + src_shape.size(),
                          dst_data + dst_shape.offset());
            }
            return;
        }

        par::copy(const_shaped_iterator<T>(src_shape, src_data, 0),
                  const_shaped_iterator<T>(src_shape, src_data, src_shape.size()),
                  dst_data + dst_shape.offset());
    } else {
        if (src_size == 1) {
            std::fill(shaped_iterator<T>(dst_shape, dst_data, 0),
                      shaped_iterator<T>(dst_shape, dst_data, dst_shape.size()),
                      src_data[src_shape.offset()]);
            return;
        }

        if (src_shape.is_contiguous()) {
            par::copy(src_data + src_shape.offset(),
                      src_data + src_shape.offset() + src_shape.size(),
                      shaped_iterator<T>(dst_shape, dst_data, 0));
            return;
        }

        par::copy(const_shaped_iterator<T>(src_shape, src_data, 0),
                  const_shaped_iterator<T>(src_shape, src_data, src_shape.size()),
                  shaped_iterator<T>(dst_shape, dst_data, 0));
    }
}

template <typename Src, typename Dst>
std::enable_if_t<is_cpu_tensor<Src>::value && is_cpu_tensor<Dst>::value>
inline reorder(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    reorder(src_shape, src.data(), src.size(), dst_shape, dst.data());
}
} // namespace impl

//==-------------------------------------------------------------------------
// DevTensor reorder operations
//==-------------------------------------------------------------------------

namespace detail {
template <typename Src, typename Dst>
std::enable_if_t<is_gpu_tensor<Src>::value && is_gpu_tensor<Dst>::value>
reorder(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    assert(src_shape == dst_shape);

    if (src_shape.is_contiguous() && dst_shape.is_contiguous() &&
        src.data() == dst.data() && src_shape.offset() == dst_shape.offset())
        return;

    if (src.shape().is_tail(src_shape) && src_shape.is_contiguous() && dst_shape.is_contiguous()) {
        gpgpu::dnn::copy(src.size(), src.data(), src_shape.offset(),
                         dst_shape.size(), dst.data(), dst_shape.offset());
    } else {
        gpgpu::dnn::copy(src_shape.size(), src_shape.extents(),
                         src.data(), src_shape.offset(), src_shape.strides(),
                         dst.data(), dst_shape.offset(), dst_shape.strides());
    }
}
} // namespace detail

//==-------------------------------------------------------------------------
// Uniform reorder operations
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

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
inline reshape(const TensorT& src, TensorT& dst) {
    if (src.size() != dst.size())
        throw shape_error("cannot reshape to destination tensor");
    flat_copy(src, dst);
}

template <typename TensorT>
enable_if_tensor<TensorT>
inline reshape(TensorT&& tensor, const std::vector<int>& new_shape) {
    tensor_type<TensorT> ret = std::forward<TensorT>(tensor);
    ret.reshape(new_shape);
    return ret;
}

template <typename TensorT>
enable_if_tensor<TensorT>
inline flatten(TensorT&& tensor, int axis) {
    tensor_type<TensorT> ret = std::forward<TensorT>(tensor);
    ret.flatten(axis);
    return ret;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
inline squeeze(TensorT&& tensor, const std::vector<int>& axes = {}) {
    tensor_type<TensorT> ret = std::forward<TensorT>(tensor);
    ret.squeeze(axes);
    return ret;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
inline unsqueeze(TensorT&& tensor, const std::vector<int>& axes) {
    tensor_type<TensorT> ret = std::forward<TensorT>(tensor);
    ret.unsqueeze(axes);
    return ret;
}

template <typename TensorT>
inline enable_if_tensor<TensorT, void>
broadcast(const TensorT& src, TensorT& dst) {
    reorder(src.broadcast(dst.shape()), dst);
}

template <typename T>
inline void transpose(const Tensor<T>& src, Tensor<T>& dst, const std::vector<size_t>& perm) {
    reorder(src.transpose(perm), dst);
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

template <typename TensorT>
enable_if_tensor<TensorT, void>
slice(const TensorT& X, TensorT& Y,
      const std::vector<int>& starts, const std::vector<int>& ends,
      const std::vector<int>& axes, const std::vector<int>& steps)
{
    Shape slice_shape = X.shape().slice(starts, ends, axes, steps);
    reorder(X, slice_shape, Y);
}

template <typename TensorT>
enable_if_tensor<TensorT, void>
slice(const TensorT& X, TensorT& Y, const std::vector<SliceDim>& dims) {
    reorder(X.slice(dims), Y);
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
flip(const TensorT& X, int axis) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("flip: invalid axis value");

    int dim = X.extent(axis);
    return X.slice({-1}, {-dim-1}, {axis}, {-1});
}

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
    auto rank = dims.size();

    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("concat: invalid axis value");

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
    cxx::conjunction<is_same_tensor<TensorT, Tensors>...>::value &&
    cxx::conjunction<std::is_same<tensor_value_type<TensorT>, tensor_value_type<Tensors>>...>::value,
    tensor_type<TensorT>
>
concat(int axis, const TensorT& first, const Tensors&... rest) {
    auto rank = first.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("concat: invalid axis value");

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
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("split: invalid axis value");

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
    auto rank = input.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("split: invalid axis value");
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

    auto rank = input.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("split: invalid axis value");

    int split_dim = input.extent(axis);
    int chunk_size = split_dim / n_split;
    int left_over = split_dim - chunk_size * n_split;
    std::vector<size_t> splits;
    for (int i = 0; i < n_split; i++)
        splits.push_back(i < left_over ? chunk_size+1 : chunk_size);
    return split(input, axis, splits);
}

} // namespace dlf
