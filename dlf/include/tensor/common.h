#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor traits
//==-------------------------------------------------------------------------

namespace detail {
template <typename TensorT>
struct tensor_traits_impl {
    using is_tensor = std::false_type;
};

template <typename T>
struct tensor_traits_impl<Tensor<T>> {
    using is_tensor = std::true_type;
    using type = Tensor<T>;
    using value_type = T;
};

template <typename T>
struct tensor_traits_impl<DevTensor<T>> {
    using is_tensor = std::true_type;
    using type = DevTensor<T>;
    using value_type = T;
};
} // namespace detail

template <typename TensorT>
struct tensor_traits : detail::tensor_traits_impl<std::decay_t<TensorT>> {};

template <typename TensorT>
using is_tensor = typename tensor_traits<TensorT>::is_tensor;

template <typename TensorT>
using tensor_type = typename tensor_traits<TensorT>::type;

template <typename TensorT>
using tensor_value_type = typename tensor_traits<TensorT>::value_type;

template <typename TensorT, typename T = tensor_type<TensorT>>
using enable_if_tensor = std::enable_if_t<is_tensor<TensorT>::value, T>;

//==-------------------------------------------------------------------------
// Tensor unary operations
//==-------------------------------------------------------------------------

#define DEFINE_UNARY_OPERATOR(name, op) \
template <typename TensorT> \
inline enable_if_tensor<TensorT> name(TensorT&& x) { \
    using T = tensor_value_type<TensorT>; \
    return ::dlf::transform(std::forward<TensorT>(x), ::dlf::xfn::op<T>()); \
}

DEFINE_UNARY_OPERATOR(abs, abs)
DEFINE_UNARY_OPERATOR(operator-, negate)
DEFINE_UNARY_OPERATOR(sign, sign)
DEFINE_UNARY_OPERATOR(reciprocal, reciprocal)
DEFINE_UNARY_OPERATOR(floor, floor)
DEFINE_UNARY_OPERATOR(ceil, ceil)
DEFINE_UNARY_OPERATOR(round, round)
DEFINE_UNARY_OPERATOR(sqrt, sqrt)
DEFINE_UNARY_OPERATOR(exp, exp)
DEFINE_UNARY_OPERATOR(log, log)
DEFINE_UNARY_OPERATOR(sin, sin)
DEFINE_UNARY_OPERATOR(cos, cos)
DEFINE_UNARY_OPERATOR(tan, tan)
DEFINE_UNARY_OPERATOR(asin, asin)
DEFINE_UNARY_OPERATOR(acos, acos)
DEFINE_UNARY_OPERATOR(atan, atan)
DEFINE_UNARY_OPERATOR(sinh, sinh)
DEFINE_UNARY_OPERATOR(cosh, cosh)
DEFINE_UNARY_OPERATOR(tanh, tanh)
DEFINE_UNARY_OPERATOR(asinh, asinh)
DEFINE_UNARY_OPERATOR(acosh, acosh)
DEFINE_UNARY_OPERATOR(atanh, atanh)
DEFINE_UNARY_OPERATOR(erf, erf)
DEFINE_UNARY_OPERATOR(sigmoid, sigmoid)
#undef DEFINE_UNARY_OPERATOR

//==-------------------------------------------------------------------------
// Tensor shape operations
//==-------------------------------------------------------------------------

template <typename TensorT>
inline enable_if_tensor<TensorT> reshape(TensorT&& tensor, const std::vector<size_t>& newshape) {
    tensor_type<TensorT> ret = std::forward<TensorT>(tensor);
    if (!ret.reshape(newshape))
        throw shape_error("cannot reshape to given shape");
    return ret;
}

template <typename T>
inline void reshape(const Tensor<T>& src, Tensor<T>& dst) {
    if (src.size() != dst.size())
        throw shape_error("cannot reshape to destination tensor");
    if (src.data() != dst.data())
        std::copy(src.begin(), src.end(), dst.begin());
}

template <typename T>
inline void reshape(const DevTensor<T>& src, DevTensor<T>& dst) {
    if (src.size() != dst.size())
        throw shape_error("cannot reshape to destination tensor");
    if (src.data() != dst.data())
        src.data().copyToAsync(gpgpu::current::queue(), dst.data(), dst.size());
}

template <typename TensorT>
inline enable_if_tensor<TensorT> flatten(TensorT&& tensor, int axis) {
    auto rank = tensor.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis > rank)
        throw shape_error("flatten: invalid axis");

    auto dims = tensor.shape().extents();
    size_t rows = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
    size_t cols = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());
    return reshape(std::forward<TensorT>(tensor), {rows, cols});
}

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
enable_if_tensor<TensorT, void>
concat(int axis, const std::vector<const tensor_type<TensorT>*>& inputs, TensorT& output) {
    auto rank = output.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("concat: invalid axis");

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

    auto dims = output.shape().extents();
    const size_t batch = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
    const size_t stride = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());
    std::vector<size_t> offsets;
    std::vector<size_t> blocks;

    size_t offset = 0;
    for (auto t : inputs) {
        auto d = t->shape().extents();
        size_t block = std::accumulate(d.begin()+axis, d.end(), 1, std::multiplies<>());
        offsets.push_back(offset);
        blocks.push_back(block);
        offset += block;
    }

    detail::concat(inputs, output, batch, stride, offsets, blocks);
}

template <typename TensorT, typename... Tensors>
std::enable_if_t<
    is_tensor<TensorT>::value &&
    cxx::conjunction<std::is_same<TensorT, Tensors>...>::value,
    tensor_type<TensorT>
>
inline concat(int axis, const TensorT& first, const Tensors&... rest) {
    auto rank = first.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("concat: invalid axis");

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
enable_if_tensor<TensorT, void>
split(int axis, const TensorT& input, const std::vector<tensor_type<TensorT>*>& outputs) {
    auto rank = input.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("split: invalid axis");

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

    auto dims = input.shape().extents();
    const size_t batch = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
    const size_t stride = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());
    std::vector<size_t> offsets;
    std::vector<size_t> blocks;

    size_t offset = 0;
    for (auto t : outputs) {
        auto d = t->shape().extents();
        size_t block = std::accumulate(d.begin()+axis, d.end(), 1, std::multiplies<>());
        offsets.push_back(offset);
        blocks.push_back(block);
        offset += block;
    }

    detail::split(input, outputs, batch, stride, offsets, blocks);
}

} // namespace dlf
