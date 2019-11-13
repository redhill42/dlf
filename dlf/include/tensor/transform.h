#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Tensor unary operations
//==-------------------------------------------------------------------------

/**
 * Transform tensor A's elements to tensor B by applying the given function.
 * The two tensor must have the same shape.
 */
template <typename TensorT, typename TensorU, typename F>
std::enable_if_t<
    is_cpu_tensor<TensorT>::value &&
    is_cpu_tensor<TensorU>::value &&
    !std::is_const<std::remove_reference_t<TensorU>>::value>
inline transformTo(const TensorT& X, TensorU&& Y, F f) {
    Y.resize(X.shape());
    map(xfn::transfer<F>(f))(Y, X);
}

/**
 * Transform a tensor to a new tensor by applying the given unary function
 * on tensor's elements.
 */
template <typename TensorT, typename F, typename U = cxx::invoke_result_t<F,tensor_value_type<TensorT>>>
std::enable_if_t<is_cpu_tensor<TensorT>::value, Tensor<U>>
inline transform(const TensorT& X, F f) {
    Tensor<U> Y{};
    transformTo(X, Y, f);
    return Y;
}

template <typename T, typename F>
std::enable_if_t<std::is_same<cxx::invoke_result_t<F,T>,T>::value, Tensor<T>>
inline transform(Tensor<T>&& A, F f) {
    map([f](auto& x){ x = f(x); })(A);
    return std::move(A);
}

//==-------------------------------------------------------------------------
// DevTensor unary operations
//==-------------------------------------------------------------------------

namespace detail {
template <typename TensorX, typename TensorY>
void gpu_transform(const std::string& name, const TensorX& X, TensorY& Y) {
    assert(X.shape() == Y.shape());
    if (X.shape().is_contiguous() && Y.shape().is_contiguous()) {
        gpgpu::dnn::transform(name, X.size(),
                              X.data(), X.shape().offset(),
                              Y.data(), Y.shape().offset());
    } else {
        gpgpu::dnn::transform(name, X.size(), X.shape().extents(),
                              X.data(), X.shape().offset(), X.shape().strides(),
                              Y.data(), Y.shape().offset(), Y.shape().strides());
    }
}

template <typename T, typename TensorX, typename TensorY>
void gpu_transform(const std::string name, const T alpha, const T beta,
                   const TensorX& X, TensorY& Y)
{
    assert(X.shape() == Y.shape());
    if (X.shape().is_contiguous() && Y.shape().is_contiguous()) {
        gpgpu::dnn::transform(name, alpha, beta, X.size(),
                              X.data(), X.shape().offset(),
                              Y.data(), Y.shape().offset());
    } else {
        gpgpu::dnn::transform(name, alpha, beta, X.size(), X.shape().extents(),
                              X.data(), X.shape().offset(), X.shape().strides(),
                              Y.data(), Y.shape().offset(), Y.shape().strides());
    }
}
} // namespace detail

template <typename TensorT, typename TensorU, typename F>
std::enable_if_t<
    is_gpu_tensor<TensorT>::value &&
    is_same_tensor<TensorT, TensorU>::value &&
    !std::is_const<std::remove_reference_t<TensorU>>::value &&
    !std::is_base_of<xfn::parameterized_function<tensor_value_type<TensorT>>, F>::value>
inline transformTo(const TensorT& X, TensorU&& Y, F) {
    Y.resize(X.shape());
    detail::gpu_transform(F::name, X, Y);
}

template <typename TensorT, typename TensorU, typename F>
std::enable_if_t<
    is_gpu_tensor<TensorT>::value &&
    is_same_tensor<TensorT, TensorU>::value &&
    !std::is_const<std::remove_reference_t<TensorU>>::value &&
    std::is_base_of<xfn::parameterized_function<tensor_value_type<TensorT>>, F>::value>
inline transformTo(const TensorT& X, TensorU&& Y, F f) {
    Y.resize(X.shape());
    detail::gpu_transform(F::name, f.alpha, f.beta, X, Y);
}

template <typename TensorT, typename F>
std::enable_if_t<is_gpu_tensor<TensorT>::value, tensor_type<TensorT>>
inline transform(const TensorT& X, F f) {
    tensor_type<TensorT> Y{};
    transformTo(X, Y, f);
    return Y;
}

template <typename T, typename Fn>
inline DevTensor<T> transform(DevTensor<T>&& X, Fn fn) {
    transformTo(X, X, fn);
    return std::move(X);
}

//==-------------------------------------------------------------------------
// Uniform tensor unary operators
//==-------------------------------------------------------------------------

#define DEFINE_UNARY_OPERATOR(name, op) \
template <typename TensorT> \
inline enable_if_tensor<TensorT> name(TensorT&& X) { \
    using T = tensor_value_type<TensorT>; \
    return transform(std::forward<TensorT>(X), xfn::op<T>()); \
}

DEFINE_UNARY_OPERATOR(abs, abs)
DEFINE_UNARY_OPERATOR(operator-, negate)
DEFINE_UNARY_OPERATOR(sign, sign)
DEFINE_UNARY_OPERATOR(reciprocal, reciprocal)
DEFINE_UNARY_OPERATOR(conj, conj)
DEFINE_UNARY_OPERATOR(floor, floor)
DEFINE_UNARY_OPERATOR(ceil, ceil)
DEFINE_UNARY_OPERATOR(round, round)
DEFINE_UNARY_OPERATOR(sqrt, sqrt)
DEFINE_UNARY_OPERATOR(square, square)
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
DEFINE_UNARY_OPERATOR(softsign, softsign)
DEFINE_UNARY_OPERATOR(softplus, softplus)
#undef DEFINE_UNARY_OPERATOR

template <typename TensorT>
inline enable_if_tensor<TensorT>
clip(TensorT&& X, const tensor_value_type<TensorT>& low, const tensor_value_type<TensorT>& high) {
    return transform(std::forward<TensorT>(X), xfn::clip<tensor_value_type<TensorT>>(low, high));
}

//==-------------------------------------------------------------------------
// Tensor binary transformation implementation
//==-------------------------------------------------------------------------

namespace detail {
template <typename T, typename U, typename IteratorC, typename F>
void transformChannel(const Shape& shape_A, const T* data_A,
                      const Shape& shape_B, const U* data_B,
                      const Shape& shape_C, IteratorC begin_C,
                      int axis, F f)
{
    assert(shape_B.rank() == 1 || shape_A.find_channel_axis(shape_B) == axis);
    assert(axis < shape_A.rank());
    assert(shape_A.extent(axis) == shape_B.size());
    assert(shape_C == shape_A);

    size_t m = 1;
    for (int i = 0; i <= axis; i++)
        m *= shape_A.extent(i);
    size_t n = shape_A.size() / m;

    tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 32, 0, n, 32), [&](auto r) {
        auto offset = r.rows().begin()*n + r.cols().begin();
        auto px = data_A + shape_A.offset() + offset;
        auto py = data_B + shape_B.offset();
        auto pz = begin_C + offset;
        for (int id = r.rows().begin(); id < r.rows().end(); id++) {
            auto y = py[id % shape_B.size()];
            std::transform(px, px+r.cols().size(), pz, [&](auto x){ return f(x, y); });
            px += n, pz += n;
        }
    });
}

template <typename T, typename U, typename W, typename F>
inline void transformChannel(const Tensor<T>& A, const Tensor<U>& B, Tensor<W>& C, size_t axis, F f) {
    transformChannel(A.shape(), A.data(), B.shape(), B.data(), C.shape(), C.begin(), axis, f);
}

template <typename LHS, typename RHS, typename RET, typename F>
std::enable_if_t<is_cpu_tensor<LHS>::value>
inline transform(const LHS& A, const RHS& B, RET& C, F f) {
    map(xfn::transfer<F>(f))(C, A, B);
}
} // namespace detail

//==-------------------------------------------------------------------------
// DevTensor binary transformation implementation
//==-------------------------------------------------------------------------

namespace detail {
template <typename T, typename R, typename F>
void transformChannel(const DevTensor<T>& A, const DevTensor<T>& B, DevTensor<R>& C, size_t axis, F) {
    assert(B.is_vector() || A.shape().find_channel_axis(B.shape()) == axis);
    assert(axis < A.rank());
    assert(A.extent(axis) == B.size());
    assert(C.shape() == A.shape());

    size_t m = A.shape().partial_size(0, axis+1);
    size_t n = A.size() / m;
    gpgpu::dnn::transform(F::name, m, n, B.size(), A.data(), 0, B.data(), 0, C.data(), 0);
}

template <typename T, typename R>
void transform(const std::string& name,
               const Shape& shape_A, const gpgpu::Buffer<T>& data_A,
               const Shape& shape_B, const gpgpu::Buffer<T>& data_B,
               const Shape& shape_C, gpgpu::Buffer<R>& data_C)
{
    if (shape_A.is_contiguous() && shape_B.is_contiguous() && shape_C.is_contiguous()) {
        if (shape_A.is_tail(shape_B) || shape_B.is_tail(shape_A)) {
            gpgpu::dnn::transform(name,
                                  shape_A.size(), data_A, shape_A.offset(),
                                  shape_B.size(), data_B, shape_B.offset(),
                                  data_C, shape_C.offset());
            return;
        }

        int axis = shape_A.find_channel_axis(shape_B);
        if (axis != -1) {
            size_t m = shape_A.partial_size(0, axis+1);
            size_t n = shape_A.size() / m;
            gpgpu::dnn::transform(name, m, n, shape_B.size(),
                                  data_A, shape_A.offset(),
                                  data_B, shape_B.offset(),
                                  data_C, shape_C.offset());
            return;
        }
    }

    auto sA = shape_A.broadcast_to(shape_C);
    auto sB = shape_B.broadcast_to(shape_C);
    gpgpu::dnn::transform(name, shape_C.size(), shape_C.extents(),
                          data_A, sA.offset(), sA.strides(),
                          data_B, sB.offset(), sB.strides(),
                          data_C, shape_C.offset(), shape_C.strides());
}

template <typename LHS, typename RHS, typename RET, typename F>
std::enable_if_t<is_gpu_tensor<LHS>::value>
inline transform(const LHS& A, const RHS& B, RET& C, F) {
    transform(F::name, A.shape(), A.data(), B.shape(), B.data(), C.shape(), C.data());
}
} // namespace detail

//==-------------------------------------------------------------------------
// Uniform tensor binary transformation
//==-------------------------------------------------------------------------

template <typename LHS, typename RHS, typename RET, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    is_same_tensor<RET,
        std::conditional_t<
            is_gpu_tensor<LHS>::value && is_relop<F>::value,
            tensor_type<LHS, bool>,
            LHS>>::value &&
    !std::is_const<std::remove_reference_t<RET>>::value>
inline transformTo(const LHS& A, const RHS& B, RET&& C, F f) {
    C.resize(Shape::broadcast(A, B));
    detail::transform(A, B, C, f);
}

template <typename LHS, typename RHS, typename RET, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    !is_tensor_view<LHS>::value &&
    !is_tensor_view<RHS>::value &&
    !is_tensor_view<RET>::value &&
    is_same_tensor<RET,
        std::conditional_t<
            is_gpu_tensor<LHS>::value && is_relop<F>::value,
            tensor_type<LHS, bool>,
            LHS>>::value>
inline transformChannel(const LHS& A, const RHS& B, RET& C, size_t axis, F f) {
    detail::transformChannel(A, B, C, axis, f);
}

#if CPP_VER >= 17
template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_same_tensor<LHS, RHS>::value, tensor_invoke_result<F, LHS, RHS>>
transform(LHS&& A, RHS&& B, F f) {
    using RET = tensor_invoke_result<F, LHS, RHS>;

    if constexpr (std::is_rvalue_reference<decltype(A)>::value &&
                  !is_tensor_view<LHS>::value &&
                  std::is_same<tensor_value_type<LHS>, tensor_value_type<RET>>::value) {
        if (A.shape() == Shape::broadcast(A, B)) {
            transformTo(A, B, A, f);
            return std::move(A);
        }
    }

    if constexpr (std::is_rvalue_reference<decltype(B)>::value &&
                  !is_tensor_view<RHS>::value &&
                  std::is_same<tensor_value_type<RHS>, tensor_value_type<RET>>::value) {
        if (B.shape() == Shape::broadcast(A, B)) {
            transformTo(A, B, B, f);
            return std::move(B);
        }
    }

    RET C;
    transformTo(A, B, C, f);
    return C;
}
#else
namespace detail {
template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_same_tensor<LHS, RHS>::value, tensor_invoke_result<F, LHS, RHS>>
basic_transform(const LHS& A, const RHS& B, F f) {
    tensor_invoke_result<F, LHS, RHS> C{};
    transformTo(A, B, C, f);
    return C;
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_same_tensor<LHS, RHS>::value, tensor_invoke_result<F, LHS, RHS>>
inline transform(const LHS& A, const RHS& B, F f) {
    return basic_transform(A, B, f);
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    !is_tensor_view<LHS>::value && !std::is_lvalue_reference<LHS>::value &&
    std::is_same<tensor_type<LHS>, tensor_invoke_result<F,LHS,RHS>>::value &&
    !is_tensor_view<RHS>::value && !std::is_lvalue_reference<RHS>::value &&
    std::is_same<tensor_type<RHS>, tensor_invoke_result<F,LHS,RHS>>::value,
    tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& A, RHS&& B, F f) {
    if (A.shape() == Shape::broadcast(A, B)) {
        transformTo(A, B, A, f);
        return std::move(A);
    }
    if (B.shape() == Shape::broadcast(A, B)) {
        transformTo(A, B, B, f);
        return std::move(B);
    }
    return basic_transform(A, B, f);
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    !is_tensor_view<LHS>::value && !std::is_lvalue_reference<LHS>::value &&
    std::is_same<tensor_type<LHS>, tensor_invoke_result<F,LHS,RHS>>::value,
    tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& A, const RHS& B, F f) {
    if (A.shape() == Shape::broadcast(A, B)) {
        transformTo(A, B, A, f);
        return std::move(A);
    }
    return basic_transform(A, B, f);
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<
    is_same_tensor<LHS, RHS>::value &&
    !is_tensor_view<RHS>::value && !std::is_lvalue_reference<RHS>::value &&
    std::is_same<tensor_type<RHS>, tensor_invoke_result<F,LHS,RHS>>::value,
    tensor_invoke_result<F, LHS, RHS>>
inline transform(const LHS& A, RHS&& B, F f) {
    if (B.shape() == Shape::broadcast(A, B)) {
        transformTo(A, B, B, f);
        return std::move(B);
    }
    return basic_transform(A, B, f);
}
} // namespace detail

template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_same_tensor<LHS, RHS>::value, tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& lhs, RHS&& rhs, F f) {
    return detail::transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), f);
}
#endif

template <typename LHS, typename RHS, typename F>
std::enable_if_t<is_tensor<LHS>::value && !is_tensor<RHS>::value, tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& A, RHS&& B, F f) {
    return transform(std::forward<LHS>(A), tensor_scalar<LHS>(std::forward<RHS>(B)), f);
}

template <typename LHS, typename RHS, typename F>
std::enable_if_t<!is_tensor<LHS>::value && is_tensor<RHS>::value, tensor_invoke_result<F, LHS, RHS>>
inline transform(LHS&& A, RHS&& B, F f) {
    return transform(tensor_scalar<RHS>(std::forward<LHS>(A)), std::forward<RHS>(B), f);
}

//==-------------------------------------------------------------------------
// Uniform tensor binary operators
//==-------------------------------------------------------------------------

#define DEFINE_BINARY_OPERATOR(op, fn)                                              \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_same_tensor<LHS, RHS>::value, LHS&>                             \
inline operator op##=(LHS& lhs, RHS&& rhs) {                                        \
    transformTo(lhs, std::forward<RHS>(rhs), lhs, xfn::fn<>());                     \
    return lhs;                                                                     \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_same_tensor<LHS, RHS>::value, LHS>                              \
inline operator op##=(LHS&& lhs, RHS&& rhs) {                                       \
    return std::move(operator op##=(lhs, std::forward<RHS>(rhs)));                  \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_tensor<LHS>::value && !is_tensor<RHS>::value, LHS&>             \
inline operator op##=(LHS& lhs, RHS&& rhs) {                                        \
    return operator op##=(lhs, tensor_scalar<LHS>(std::forward<RHS>(rhs)));         \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_tensor<LHS>::value && !is_tensor<RHS>::value, LHS>              \
inline operator op##=(LHS&& lhs, RHS&& rhs) {                                       \
    return std::move(operator op##=(lhs, std::forward<RHS>(rhs)));                  \
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
std::enable_if_t<
    is_cpu_tensor<LHS>::value && is_cpu_tensor<RHS>::value &&
    is_exactly_same_tensor<LHS, RHS>::value,
    bool>
inline operator==(const LHS& lhs, const RHS& rhs) {
    return lhs.shape() == rhs.shape() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename LHS, typename RHS>
std::enable_if_t<
    is_cpu_tensor<LHS>::value && is_cpu_tensor<RHS>::value &&
    is_exactly_same_tensor<LHS, RHS>::value,
    bool>
inline operator!=(const LHS& lhs, const RHS& rhs) {
    return !(lhs == rhs);
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::equal_to<>>
inline equal(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::equal_to<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::not_equal_to<>>
inline not_equal(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::not_equal_to<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::less<>>
inline operator<(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::less<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::less_equal<>>
inline operator<=(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::less_equal<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::greater<>>
inline operator>(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::greater<>());
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::greater_equal<>>
inline operator>=(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::greater_equal<>());
}

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
    return transform(std::forward<TensorT>(x), xfn::logical_not<tensor_value_type<TensorT>>());
}

template <typename LHS, typename RHS>
inline enable_if_tensors<LHS, RHS, xfn::power<>>
pow(LHS&& lhs, RHS&& rhs) {
    return transform(std::forward<LHS>(lhs), std::forward<RHS>(rhs), xfn::power<>());
}

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
std::enable_if_t<!is_tensor_view<TensorA>::value>
inline aggregate(Fn, TensorY& Y, TensorA&& A) {
    assert(A.shape() == Y.shape());
    flat_copy(std::forward<TensorA>(A), Y);
}

template <typename Fn, typename TensorY, typename TensorA>
std::enable_if_t<is_tensor_view<TensorA>::value>
inline aggregate(Fn, TensorY& Y, TensorA&& A) {
    assert(A.shape() == Y.shape());
    reorder(std::forward<TensorA>(A), Y);
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
// Tensor ternary operations
//==-------------------------------------------------------------------------

template <typename TensorK, typename TensorX, typename TensorY>
std::enable_if_t<
    is_cpu_tensor<TensorX>::value &&
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorK, tensor_type<TensorX, bool>>::value>
inline where(const TensorK& K, const TensorX& X, const TensorY& Y, tensor_type<TensorX>& Z) {
    Z.resize(Shape::broadcast(K, X, Y));
    map([](auto k, auto x, auto y, auto& z) {
        z = k ? x : y;
    })(K, X, Y, Z);
}

template <typename TensorK, typename TensorX, typename TensorY>
std::enable_if_t<
    is_gpu_tensor<TensorX>::value &&
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorK, tensor_type<TensorX, bool>>::value>
inline where(const TensorK& K, const TensorX& X, const TensorY& Y, tensor_type<TensorX>& Z) {
    auto final_shape = Shape::broadcast(K, X, Y);
    Z.resize(final_shape);

    auto k_shape = K.shape().broadcast_to(final_shape);
    auto x_shape = X.shape().broadcast_to(final_shape);
    auto y_shape = Y.shape().broadcast_to(final_shape);

    gpgpu::dnn::where(
        final_shape.size(), final_shape.extents(),
        K.data(), k_shape.offset(), k_shape.strides(),
        X.data(), x_shape.offset(), x_shape.strides(),
        Y.data(), y_shape.offset(), y_shape.strides(),
        Z.data(), 0);
}

template <typename TensorK, typename TensorX, typename TensorY>
std::enable_if_t<
    is_exactly_same_tensor<TensorX, TensorY>::value &&
    is_exactly_same_tensor<TensorK, tensor_type<TensorX, bool>>::value,
    tensor_type<TensorX>>
where(const TensorK& K, const TensorX& X, const TensorY& Y) {
    tensor_type<TensorX> Z{};
    where(K, X, Y, Z);
    return Z;
}

template <typename TensorI, typename TensorV, typename TensorR>
std::enable_if_t<
    is_cpu_tensor<TensorI>::value && is_cpu_tensor<TensorV>::value &&
    is_exactly_same_tensor<TensorV, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
one_hot(const TensorI& indices, const TensorV& values, TensorR&& output, size_t depth, int axis = -1) {
    assert(values.rank() == 1 && values.extent(0) == 2);

    auto rank = indices.rank() + 1;
    detail::norm_axis(rank, axis);

    std::vector<size_t> depth_dims(rank, 1);
    depth_dims[axis] = depth;

    auto target = Tensor<int>(Shape(depth_dims)).range(0);
    transformTo(unsqueeze(indices, axis), target, output,
                [depth, on=values(1), off=values(0)](auto x, auto b) {
                    auto a = static_cast<int>(x);
                    if (a < 0) a += depth;
                    return a == b ? on : off;
                });
}

template <typename TensorI, typename TensorV, typename TensorR>
std::enable_if_t<
    is_gpu_tensor<TensorI>::value && !is_tensor_view<TensorI>::value &&
    is_gpu_tensor<TensorV>::value && !is_tensor_view<TensorV>::value &&
    is_gpu_tensor<TensorR>::value && !is_tensor_view<TensorR>::value &&
    is_same_tensor<TensorI, TensorV>::value &&
    is_same_tensor<TensorV, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
one_hot(const TensorI& indices, const TensorV& values, TensorR&& output, size_t depth, int axis = -1) {
    assert(values.rank() == 1 && values.extent(0) == 2);

    auto rank = indices.rank() + 1;
    detail::norm_axis(rank, axis);

    auto output_dims = indices.shape().extents();
    output_dims.insert(output_dims.begin() + axis, depth);
    output.resize(Shape(output_dims));

    auto k = indices.shape().partial_size(axis, indices.rank());
    gpgpu::dnn::onehot(output.size(), depth, k, indices.data(), values.data(), output.data());
}

} // namespace dlf
