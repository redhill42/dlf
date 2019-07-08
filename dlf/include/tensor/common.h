#pragma once

#include <unordered_set>

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor traits
//==-------------------------------------------------------------------------

namespace detail {
template <typename T> struct cpu {};
template <typename T> struct gpu {};

template <typename T>
struct tensor_traits_impl {
    using is_tensor = std::false_type;
    using tag = void;
    using value_type = T;

    template <typename U>
    using tensor_type = void;
};

template <typename T>
struct tensor_traits_impl<Tensor<T>> {
    using is_tensor = std::true_type;
    using tag = cpu<void>;
    using value_type = T;

    template <typename U>
    using tensor_type = Tensor<std::decay_t<U>>;

    template <typename U>
    static tensor_type<U> scalar(U&& value) {
        tensor_type<U> ret({1});
        ret.data()[0] = std::forward<U>(value);
        return ret;
    }
};

template <typename T>
struct tensor_traits_impl<DevTensor<T>> {
    using is_tensor = std::true_type;
    using tag = gpu<T>;
    using value_type = T;

    template <typename U>
    using tensor_type = DevTensor<std::decay_t<U>>;

    template <typename U>
    static tensor_type<U> scalar(U value) {
        tensor_type<U> ret({1});
        ret.data().write(gpgpu::current::queue(), &value, 1);
        return ret;
    }
};
} // namespace detail

template <typename TensorT>
struct tensor_traits : detail::tensor_traits_impl<std::decay_t<TensorT>> {};

template <typename TensorT>
using is_tensor = typename tensor_traits<TensorT>::is_tensor;

template <typename X, typename Y>
using is_same_tensor = cxx::conjunction<
    is_tensor<X>, is_tensor<Y>,
    std::is_same<typename tensor_traits<X>::tag, typename tensor_traits<Y>::tag>>;

template <typename TensorT>
using tensor_value_type = typename tensor_traits<TensorT>::value_type;

template <typename TensorT, typename U = tensor_value_type<TensorT>>
using tensor_type = typename tensor_traits<TensorT>::template tensor_type<U>;

template <typename TensorT, typename R = tensor_type<TensorT>>
using enable_if_tensor = std::enable_if_t<is_tensor<TensorT>::value, R>;

template <typename Fn, typename LHS, typename RHS>
using tensor_invoke_result =
    std::conditional_t<is_tensor<LHS>::value,
        tensor_type<LHS, cxx::invoke_result_t<Fn, tensor_value_type<LHS>, tensor_value_type<RHS>>>,
        tensor_type<RHS, cxx::invoke_result_t<Fn, tensor_value_type<LHS>, tensor_value_type<RHS>>>>;

template <typename LHS, typename RHS, typename Fn, typename R = tensor_invoke_result<Fn, LHS, RHS>>
using enable_if_tensors =
    std::enable_if_t<
        is_same_tensor<LHS, RHS>::value ||
        is_tensor<LHS>::value ||
        is_tensor<RHS>::value, R>;

template <typename TensorT, typename U>
inline tensor_type<TensorT, U> tensor_scalar(U&& value) {
    return tensor_traits<TensorT>::scalar(std::forward<U>(value));
}

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
// Tensor binary operations
//==-------------------------------------------------------------------------

template <typename LHS, typename RHS, typename Fn>
std::enable_if_t<is_tensor<LHS>::value && !is_tensor<RHS>::value, tensor_invoke_result<Fn, LHS, RHS>>
inline transform(LHS&& lhs, RHS&& rhs, Fn fn) {
    return transform(std::forward<LHS>(lhs), tensor_scalar<LHS>(std::forward<RHS>(rhs)), fn);
}

template <typename LHS, typename RHS, typename Fn>
std::enable_if_t<!is_tensor<LHS>::value && is_tensor<RHS>::value, tensor_invoke_result<Fn, LHS, RHS>>
inline transform(LHS&& lhs, RHS&& rhs, Fn fn) {
    return transform(tensor_scalar<RHS>(std::forward<LHS>(lhs)), std::forward<RHS>(rhs), fn);
}

#define DEFINE_BINARY_OPERATOR(op, fn)                                              \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_same_tensor<LHS, RHS>::value, tensor_type<LHS>&>                \
inline operator op##=(LHS& lhs, RHS&& rhs) {                                        \
    return transformTo(lhs, std::forward<RHS>(rhs), lhs, xfn::fn<>());              \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_tensor<LHS>::value && !is_tensor<RHS>::value, tensor_type<LHS>&>\
inline operator op##=(LHS& lhs, RHS&& rhs) {                                        \
    return operator op##=(lhs, tensor_scalar<LHS>(std::forward<RHS>(rhs)));         \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
enable_if_tensors<LHS, RHS, xfn::fn<>>                                              \
inline operator op(LHS&& lhs, RHS&& rhs) {                                          \
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::fn<>());  \
}

DEFINE_BINARY_OPERATOR(+, plus)
DEFINE_BINARY_OPERATOR(-, minus)
DEFINE_BINARY_OPERATOR(*, multiplies)
DEFINE_BINARY_OPERATOR(/, divides)
DEFINE_BINARY_OPERATOR(%, modulus)
DEFINE_BINARY_OPERATOR(&, bit_and)
DEFINE_BINARY_OPERATOR(|, bit_or)
DEFINE_BINARY_OPERATOR(^, bit_xor)
#undef DEFINE_BINARY_OPERATOR

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::logical_and<>>
inline operator&&(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::logical_and<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::logical_or<>>
inline operator||(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::logical_or<>());
}

template <typename TensorT>
enable_if_tensor<TensorT, Tensor<bool>>
inline operator!(TensorT&& x) {
    return transform(x, xfn::logical_not<tensor_value_type<TensorT>>());
}

template <typename LHS, typename RHS>
inline enable_if_tensors<LHS, RHS, xfn::power<>>
pow(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::power<>());
}

template <typename TensorT>
enable_if_tensor<TensorT> matpow(TensorT&& x, long n) {
    assert(x.is_square() && n >= 0);
    if (n == 0)
        return Tensor<tensor_value_type<TensorT>>::identity(x.extent(0));
    n--;

    auto A = x, B = x, t = x;
    while (n > 0) {
        if (n & 1)
            std::swap(B, dot(A, B, &t));
        std::swap(A, dot(A, A, &t));
        n >>= 1;
    }
    return B;
}

namespace dot_product {
/**
 * We use comma operator to represent dot product, because C++ doesn't have dot
 * operator yet, and comma and dot are looks similar. To use the comma operator
 * be sure to enclose the expression in parentheses to avoid ambiguity. That is,
 * use
 *     auto z = (x , y)
 * instead of
 *     auto z = x, y
 */
template <typename T>
Tensor<T> operator,(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return dot(lhs, rhs);
}

template <typename T>
inline DevTensor<T> operator,(const DevTensor<T>& lhs, const DevTensor<T>& rhs) {
    return dot(lhs, rhs);
}
} // namespace dot_product

//==-------------------------------------------------------------------------
// Tensor aggregate operations
//==-------------------------------------------------------------------------

namespace detail {
template <typename Fn, typename... Tensors>
struct aggregate_type {};

template <typename Fn, typename TensorT>
struct aggregate_type<Fn, TensorT> {
    using type = tensor_type<TensorT>;
};

template <typename Fn, typename First, typename... Tensors>
struct aggregate_type<Fn, First, Tensors...> {
    using type = tensor_invoke_result<Fn, First, typename aggregate_type<Fn, Tensors...>::type>;
};

template <typename Fn, typename TensorY, typename TensorA>
inline void aggregate(Fn, TensorY& Y, TensorA&& A) {
    copy(std::forward<TensorA>(A), Y);
}

template <typename Fn, typename TensorY, typename First, typename Second, typename... Rest>
inline void aggregate(Fn fn, TensorY& Y, First&& first, Second&& second, Rest... rest) {
    transformTo(std::forward<First>(first), std::forward<Second>(second), Y, fn);
    aggregate(fn, Y, Y, rest...);
}
} // namespace detail

template <typename Fn, typename First, typename... Rest,
    typename = std::enable_if_t<
        is_tensor<First>::value &&
        cxx::conjunction<is_same_tensor<First, Rest>...>::value>>
auto aggregate(Fn fn, First&& first, Rest&&... rest)
    -> typename detail::aggregate_type<Fn, First, Rest...>::type
{
    using TensorR = typename detail::aggregate_type<Fn, First, Rest...>::type;
    TensorR result(Shape::broadcast(first, rest...));
    detail::aggregate(fn, result, std::forward<First>(first), std::forward<Rest>(rest)...);
    return result;
}

template <typename First, typename... Rest>
inline auto max(First&& first, Rest&&... rest) {
    return aggregate(xfn::max<tensor_value_type<First>>(),
                    std::forward<First>(first),
                    std::forward<Rest>(rest)...);
}

template <typename First, typename... Rest>
inline auto min(First&& first, Rest&&... rest) {
    return aggregate(xfn::min<tensor_value_type<First>>(),
                     std::forward<First>(first),
                     std::forward<Rest>(rest)...);
}

template <typename First, typename... Rest>
inline auto sum(First&& first, Rest&&... rest) {
    return aggregate(xfn::plus<>(), std::forward<First>(first), std::forward<Rest>(rest)...);
}

template <typename First, typename... Rest>
inline auto mean(First&& first, Rest&&... rest) {
    auto result = sum(first, rest...);
    using T = tensor_value_type<decltype(result)>;
    result /= static_cast<T>(1 + sizeof...(rest));
    return result;
}

template <typename First, typename... Rest>
inline auto product(First&& first, Rest&&... rest) {
    return aggregate(xfn::multiplies<>(), std::forward<First>(first), std::forward<Rest>(rest)...);
}

//==-------------------------------------------------------------------------
// Tensor shape operations
//==-------------------------------------------------------------------------

template <typename TensorT>
inline enable_if_tensor<TensorT> broadcast(TensorT&& src, const Shape& shape) {
    if (src.shape() == shape) {
        return std::forward<TensorT>(src);
    } else {
        tensor_type<TensorT> dst(shape);
        broadcast(src, dst);
        return dst;
    }
}

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
enable_if_tensor<TensorT> flatten(TensorT&& tensor, int axis) {
    auto rank = tensor.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis > rank)
        throw shape_error("flatten: invalid axis value");

    auto dims = tensor.shape().extents();
    size_t rows = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
    size_t cols = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());
    return reshape(std::forward<TensorT>(tensor), {rows, cols});
}

template <typename TensorT>
enable_if_tensor<TensorT> squeeze(TensorT&& tensor, const std::vector<int>& axes = {}) {
    auto rank = tensor.rank();

    std::unordered_set<int> adjusted_axes;
    for (auto a : axes) {
        if (a < 0) a += rank;
        if (a < 0 || a >= rank)
            throw shape_error("squeeze: invalid axis value");
        adjusted_axes.insert(a); // duplicate is ok
    }

    std::vector<size_t> shape;
    for (int i = 0; i < rank; i++) {
        auto dim = tensor.extent(i);
        if (adjusted_axes.find(i) != adjusted_axes.end()) {
            if (dim != 1)
                throw shape_error("squeeze: cannot select an axis to squeeze out which has size not equal to 1");
            continue;
        } else if (adjusted_axes.empty() && dim == 1) {
            continue;
        } else {
            shape.push_back(dim);
        }
    }

    return reshape(std::forward<TensorT>(tensor), shape);
}

template <typename TensorT>
enable_if_tensor<TensorT> unsqueeze(TensorT&& tensor, const std::vector<int>& axes) {
    auto rank = tensor.rank() + axes.size();
    std::unordered_set<int> adjusted_axes;
    for (auto a : axes) {
        if (a < 0) a += rank;
        if (a < 0 || a >= rank)
            throw shape_error("unsqueeze: invalid axis value");
        if (adjusted_axes.find(a) != adjusted_axes.end())
            throw shape_error("unsqueeze: duplicate axis value");
        adjusted_axes.insert(a);
    }

    std::vector<size_t> shape;
    for (size_t i = 0, j = 0; i < rank; i++) {
        if (adjusted_axes.find(i) != adjusted_axes.end()) {
            shape.push_back(1);
        } else {
            shape.push_back(tensor.extent(j++));
        }
    }

    return reshape(std::forward<TensorT>(tensor), shape);
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

    auto dims = output.shape().extents();
    const size_t batch = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
    const size_t stride = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());
    std::vector<size_t> offsets, blocks;

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
enable_if_tensor<TensorT, void>
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

    auto dims = input.shape().extents();
    const size_t batch = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
    const size_t stride = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());
    std::vector<size_t> offsets, blocks;

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

namespace detail {
template <typename T>
inline void transpose(const Tensor<T>& src, const Shape& shape, Tensor<T>& dst) {
    copy(src, shape, dst);
}

template <typename T>
inline void transpose(const DevTensor<T>& src, const Shape& shape, DevTensor<T>& dst) {
    if (shape.rank() == 2 && !src.shape().is_identical(shape)) {
        gpgpu::blas::omatcopy(gpgpu::blas::Layout::RowMajor,
                              gpgpu::blas::Transpose::Trans,
                              src.extent(0), src.extent(1),
                              T(1), src.data(), src.stride(0),
                              dst.data(), dst.stride(0));
    } else {
        copy(src, shape, dst);
    }
}
} // namespace detail

template <typename TensorT>
enable_if_tensor<TensorT, void>
transpose(const TensorT& src, const std::vector<size_t> perm, TensorT& dst) {
    Shape shape = src.shape().transpose(perm);
    if (shape != dst.shape())
        throw shape_error("transpose: invalid output shape");
    detail::transpose(src, shape, dst);
}

template <typename TensorT>
enable_if_tensor<TensorT> transpose(const TensorT& src, const std::vector<size_t>& perm) {
    tensor_type<TensorT> dst(src.shape().transpose(perm));
    transpose(src, perm, dst);
    return dst;
}

template <typename TensorT>
enable_if_tensor<TensorT> transpose(TensorT&& src) {
    if (src.is_vector()) {
        return reshape(std::forward<TensorT>(src), {src.extent(0), 1});
    } else {
        std::vector<size_t> perm(src.rank());
        std::iota(perm.begin(), perm.end(), 0);
        std::reverse(perm.begin(), perm.end());
        return transpose(src, perm);
    }
}

/**
 * We use ~ operator to represent tensor transposition instead of bitwise not
 * operator.
 */
template <typename TensorT>
inline enable_if_tensor<TensorT> operator~(TensorT&& src) {
    return transpose(std::forward<TensorT>(src));
}

} // namespace dlf
