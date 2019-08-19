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
                         dst.size(), dst.data(), dst_shape.offset());
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

//==-------------------------------------------------------------------------

namespace detail {
template <typename T>
void concat(const std::vector<const Tensor<T>*>& inputs,
            Tensor<T>& output,
            size_t batch, size_t stride,
            const std::vector<size_t>& offsets,
            const std::vector<size_t>& blocks)
{
    tbb::parallel_for(tbb::blocked_range2d<int>(0, inputs.size(), 1, 0, batch, 256), [&](auto r) {
        for (int i = r.rows().begin(); i < r.rows().end(); i++) {
            auto& t = *inputs[i];
            for (int j = r.cols().begin(); j < r.cols().end(); j++) {
                std::copy(t.data() + blocks[i]*j,
                          t.data() + blocks[i]*(j+1),
                          output.data() + stride*j + offsets[i]);
            }
        }
    });
}

template <typename T>
void concat(const std::vector<const DevTensor<T>*>& inputs,
            DevTensor<T>& output,
            size_t, size_t stride,
            const std::vector<size_t>& offsets,
            const std::vector<size_t>& blocks)
{
    for (size_t i = 0; i < inputs.size(); i++) {
        auto& t = *inputs[i];
        gpgpu::dnn::concat_copy(t.size(), offsets[i], blocks[i], stride, t.data(), output.data());
    }
}

template <typename T>
void split(const Tensor<T>& input,
           const std::vector<Tensor<T>*>& outputs,
           size_t batch, size_t stride,
           const std::vector<size_t>& offsets,
           const std::vector<size_t>& blocks)
{
    tbb::parallel_for(tbb::blocked_range2d<int>(0, outputs.size(), 1, 0, batch, 256), [&](auto r) {
        for (int i = r.rows().begin(); i < r.rows().end(); i++) {
            auto& t = *outputs[i];
            for (int j = r.cols().begin(); j < r.cols().end(); j++) {
                std::copy(input.data() + stride*j + offsets[i],
                          input.data() + stride*j + offsets[i] + blocks[i],
                          t.data() + blocks[i]*j);
            }
        }
    });
}

template <typename T>
void split(const DevTensor<T>& input,
           const std::vector<DevTensor<T>*>& outputs,
           size_t, size_t stride,
           const std::vector<size_t>& offsets,
           const std::vector<size_t>& blocks)
{
    for (size_t i = 0; i < outputs.size(); i++) {
        auto& t = *outputs[i];
        gpgpu::dnn::split_copy(t.size(), offsets[i], blocks[i], stride, input.data(), t.data());
    }
}
} // namespace detail

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
concat(int axis, const std::vector<const tensor_type<TensorT>*>& inputs, TensorT& output) {
    auto rank = output.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("concat: invalid axis value");

    size_t dim_sum = 0;
    for (auto t : inputs) {
        if (t->rank() != rank)
            throw shape_error("concat: all tensors to concat must have same rank");
        for (size_t i = 0; i < rank; i++) {
            if (i == axis) {
                dim_sum += t->extent(i);
            } else if (t->extent(i) != output.extent(i)) {
                throw shape_error("concat: incompatible input tensor shape");
            }
        }
    }
    if (dim_sum != output.extent(axis)) {
        throw shape_error("concat: incompatible input tensor shape");
    }

    const size_t batch = output.shape().partial_size(0, axis);
    const size_t stride = output.shape().partial_size(axis, output.rank());
    std::vector<size_t> offsets, blocks;

    size_t offset = 0;
    for (auto t : inputs) {
        size_t block = t->shape().partial_size(axis, t->rank());
        offsets.push_back(offset);
        blocks.push_back(block);
        offset += block;
    }

    detail::concat(inputs, output, batch, stride, offsets, blocks);
}

template <typename TensorT, typename... Tensors>
std::enable_if_t<
    is_tensor<TensorT>::value && !is_tensor_view<TensorT>::value &&
    cxx::conjunction<std::is_same<TensorT, Tensors>...>::value,
    tensor_type<TensorT>
>
concat(int axis, const TensorT& first, const Tensors&... rest) {
    auto rank = first.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("concat: invalid axis value");

    std::vector<const tensor_type<TensorT>*> inputs{&first, &rest...};
    std::vector<size_t> dims = first.shape().extents();
    dims[axis] = 0;
    for (auto t : inputs) {
        if (t->rank() != rank)
            throw shape_error("concat: all tensors to concat must have same rank");
        dims[axis] += t->extent(axis);
    }

    tensor_type<TensorT> res{Shape(dims)};
    concat(axis, inputs, res);
    return res;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
split(int axis, const TensorT& input, const std::vector<tensor_type<TensorT>*>& outputs) {
    auto rank = input.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("split: invalid axis value");

    size_t dim_sum = 0;
    for(auto t : outputs) {
        if (t->rank() != rank)
            throw shape_error("split: all output tensors must have same rank");
        for (size_t i = 0; i < rank; i++) {
            if (i == axis) {
                dim_sum += t->extent(axis);
            } else if (t->extent(i) != input.extent(i)) {
                throw shape_error("split: incompatible output tensor shape");
            }
        }
    }
    if (dim_sum != input.extent(axis)) {
        throw shape_error("split: incompatible output tensor shape");
    }

    const size_t batch = input.shape().partial_size(0, axis);
    const size_t stride = input.shape().partial_size(axis, input.rank());
    std::vector<size_t> offsets, blocks;

    size_t offset = 0;
    for (auto t : outputs) {
        size_t block = t->shape().partial_size(axis, t->rank());
        offsets.push_back(offset);
        blocks.push_back(block);
        offset += block;
    }

    detail::split(input, outputs, batch, stride, offsets, blocks);
}

} // namespace dlf
