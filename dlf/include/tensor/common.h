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
    using is_view = std::false_type;
    using tag = void;
    using value_type = T;

    template <typename U>
    using tensor_type = void;

    template <typename U>
    using tensor_view_type = void;
};

template <typename T>
struct tensor_traits_impl<Tensor<T>> {
    using is_tensor = std::true_type;
    using is_view = std::false_type;
    using tag = cpu<void>;
    using value_type = T;

    template <typename U>
    using tensor_type = Tensor<std::decay_t<U>>;

    template <typename U>
    using tensor_view_type = TensorView<std::decay_t<U>>;
};

template <typename T>
struct tensor_traits_impl<TensorView<T>> {
    using is_tensor = std::true_type;
    using is_view = std::true_type;
    using tag = cpu<void>;
    using value_type = T;

    template <typename U>
    using tensor_type = Tensor<std::decay_t<U>>;

    template <typename U>
    using tensor_view_type = TensorView<std::decay_t<U>>;
};

template <typename T>
struct tensor_traits_impl<DevTensor<T>> {
    using is_tensor = std::true_type;
    using is_view = std::false_type;
    using tag = gpu<T>;
    using value_type = T;

    template <typename U>
    using tensor_type = DevTensor<std::decay_t<U>>;

    template <typename U>
    using tensor_view_type = DevTensorView<std::decay_t<U>>;
};

template <typename T>
struct tensor_traits_impl<DevTensorView<T>> {
    using is_tensor = std::true_type;
    using is_view = std::true_type;
    using tag = gpu<T>;
    using value_type = T;

    template <typename U>
    using tensor_type = DevTensor<std::decay_t<U>>;

    template <typename U>
    using tensor_view_type = DevTensorView<std::decay_t<U>>;
};
} // namespace detail

template <typename TensorT>
struct tensor_traits : detail::tensor_traits_impl<std::decay_t<TensorT>> {};

template <typename TensorT>
using is_tensor = typename tensor_traits<TensorT>::is_tensor;

template <typename TensorT>
using is_tensor_view = typename tensor_traits<TensorT>::is_view;

template <typename X, typename Y>
using is_same_tensor = cxx::conjunction<
    is_tensor<X>, is_tensor<Y>,
    std::is_same<typename tensor_traits<X>::tag, typename tensor_traits<Y>::tag>>;

template <typename TensorT>
using tensor_value_type = typename tensor_traits<TensorT>::value_type;

template <typename TensorT, typename U = tensor_value_type<TensorT>>
using tensor_type = typename tensor_traits<TensorT>::template tensor_type<U>;

template <typename TensorT, typename U = tensor_value_type<TensorT>>
using tensor_view_type = typename tensor_traits<TensorT>::template tensor_view_type<U>;

template <typename TensorT, typename R = tensor_type<TensorT>>
using enable_if_tensor = std::enable_if_t<is_tensor<TensorT>::value, R>;

template <typename TensorT, typename R = tensor_type<TensorT>>
using enable_if_non_view_tensor =
    std::enable_if_t<is_tensor<TensorT>::value && !is_tensor_view<TensorT>::value, R>;

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

template <typename LHS, typename RHS, typename Fn, typename R = tensor_invoke_result<Fn, LHS, RHS>>
using enable_if_non_view_tensors =
    std::enable_if_t<
        !is_tensor_view<LHS>::value &&
        !is_tensor_view<RHS>::value &&
        (is_same_tensor<LHS, RHS>::value ||
         is_tensor<LHS>::value ||
         is_tensor<RHS>::value),
        R>;

template <typename TensorT, typename U>
inline tensor_type<TensorT, U> tensor_scalar(U&& value) {
    return tensor_type<TensorT, U>::scalar(std::forward<U>(value));
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
std::enable_if_t<is_same_tensor<LHS, RHS>::value, LHS&>                             \
inline operator op##=(LHS& lhs, RHS&& rhs) {                                        \
    return transformTo(lhs, std::forward<RHS>(rhs), lhs, xfn::fn<>());              \
}                                                                                   \
                                                                                    \
template <typename LHS, typename RHS>                                               \
std::enable_if_t<is_tensor<LHS>::value && !is_tensor<RHS>::value, LHS&>             \
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

// vector or matrix dot product.

template <typename TensorT>
inline enable_if_non_view_tensor<TensorT> dot(const TensorT& A, const TensorT& B) {
    TensorT C;
    dot(A, B, &C);
    return C;
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
inline Tensor<T> operator,(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    return dot(lhs, rhs);
}

template <typename T>
inline DevTensor<T> operator,(const DevTensor<T>& lhs, const DevTensor<T>& rhs) {
    return dot(lhs, rhs);
}
} // namespace dot_product

/**
 * The cross product on tensors is typically referred to as the tensor product.
 * Given a tensor a of order q with dimensions (i1, ..., iq), and a tensor b
 * of order r with dimensions (j1, ..., jr), their cross product c is of order
 * q + r with dimensions (k1, ..., kq+r) which are the i dimensions followed
 * by the j dimensions.
 */
template <typename LHS, typename RHS, typename Fn>
enable_if_tensors<LHS, RHS, Fn>
cross(const LHS& A, const RHS& B, Fn f) {
    std::vector<int> axesA(B.rank()), axesB(A.rank());
    std::iota(axesA.begin(), axesA.end(), A.rank()); // unsqueeze right
    std::iota(axesB.begin(), axesB.end(), 0);        // unsqueeze left
    return transform(tensor_view_type<LHS>(A.shape().unsqueeze(axesA), A),
                     tensor_view_type<RHS>(B.shape().unsqueeze(axesB), B),
                     f);
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::multiplies<>>
cross(const LHS& lhs, const RHS& rhs) {
    return cross(lhs, rhs, xfn::multiplies<>());
}

// General matrix multiplication

template <typename TensorT>
inline enable_if_non_view_tensor<TensorT, void> gemm(
    const tensor_value_type<TensorT>& alpha, const TensorT& A, const TensorT& B,
    const tensor_value_type<TensorT>& beta, const TensorT& C, TensorT& Y,
    bool transA = false, bool transB = false, TensorT* work = nullptr)
{
    broadcast(C, Y);
    gemm(alpha, A, B, beta, &Y, transA, transB, work);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT> gemm(
    const tensor_value_type<TensorT>& alpha, const TensorT& A, const TensorT& B,
    const tensor_value_type<TensorT>& beta, const TensorT& C,
    bool transA = false, bool transB = false, TensorT* work = nullptr)
{
    assert(A.is_matrix() && B.is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    assert(k == p);

    tensor_type<TensorT> Y = C.broadcast({m, n});
    gemm(alpha, A, B, beta, &Y, transA, transB, work);
    return Y;
}

namespace detail {
inline int matmul_broadcast(Shape& shapeA, Shape& shapeB, Shape& shapeC) {
    if (shapeA.rank() == 0 || shapeB.rank() == 0)
        throw shape_error("matmul: Input tensors of wrong rank (0).");

    // First promote each shape to at least rank-2. This logic is
    // specific to matmul, not generic broadcasting.
    if (shapeA.rank() == 1)
        shapeA = shapeA.unsqueeze(0);
    if (shapeB.rank() == 1)
        shapeB = shapeB.unsqueeze(1);

    auto dimsA = shapeA.extents();
    auto dimsB = shapeB.extents();

    // Check for compatible matrix multiply dimensions
    auto m = dimsA[dimsA.size() - 2];
    auto k = dimsA[dimsA.size() - 1];
    auto p = dimsB[dimsB.size() - 2];
    auto n = dimsB[dimsB.size() - 1];

    if (k != p)
        throw shape_error("matmul: Incompatible dimensions for matrix multiplication");

    // Now call out to generic multidimensional broadcasting for
    // the broadcastable prefixes.
    auto prefixShape = Shape::broadcast(
        Shape(std::vector<size_t>{dimsA.begin(), dimsA.end() - 2}),
        Shape(std::vector<size_t>{dimsB.begin(), dimsB.end() - 2})
    );

    // Back to matmul-specific. Add the trailing dimensions back in.
    dimsA = prefixShape.extents();
    dimsA.push_back(m);
    dimsA.push_back(k);
    shapeA = shapeA.broadcast(Shape(dimsA));

    dimsB = prefixShape.extents();
    dimsB.push_back(k);
    dimsB.push_back(n);
    shapeB = shapeB.broadcast(Shape(dimsB));

    auto dimsC = prefixShape.extents();
    dimsC.push_back(m);
    dimsC.push_back(n);
    shapeC = Shape(dimsC);

    return prefixShape.size();
}
} // namespace detail

/**
 * Matrix product of two arrays.
 *
 * The behavior depends on the arguments in the following way.
 *
 *  - If both arguments are 2-D they are multiplied like conventional matrices.
 *  - If either argument is N-D, N > 2, it is treated as a stack of matrices
 *    residing in the last two indexes and broadcast accordingly.
 *  - If the first argument is 1-D, it is promoted to a matrix by prepending
 *    a 1 to its dimensions. After matrix multiplication the prepended 1 is
 *    removed.
 *  - If the second argument is 1-D, it is promoted to a matrix by appending
 *    a 1 to its dimensions. After matrix multiplication the appended 1 is
 *    removed.
 *
 * Multiplication by a scalar is not allowed, use * instead. Note that
 * multiplying a stack of matrices with a vector will result in a stack of
 * vectors, but matmul will not recognize it as such.
 */
template <typename T>
void matmul(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& C) {
    if (A.rank() <= 2 && B.rank() <= 2) {
        dot(A, B, &C);
        return;
    }

    Shape shapeA = A.shape();
    Shape shapeB = B.shape();
    Shape shapeC;

    int batch = detail::matmul_broadcast(shapeA, shapeB, shapeC);

    int m = shapeA.extent(shapeA.rank() - 2);
    int k = shapeA.extent(shapeA.rank() - 1);
    int n = shapeB.extent(shapeB.rank() - 1);
    int lda = std::max<int>(shapeA.stride(shapeA.rank() - 2), k);
    int ldb = std::max<int>(shapeB.stride(shapeB.rank() - 2), n);
    int ldc = std::max<int>(shapeC.stride(shapeC.rank() - 2), n);
    int off_a = shapeA.stride(shapeA.rank() - 3);
    int off_b = shapeB.stride(shapeB.rank() - 3);
    int off_c = shapeC.stride(shapeC.rank() - 3);

    if (A.rank() == 1)
        shapeC = shapeC.squeeze(-2);
    if (B.rank() == 1)
        shapeC = shapeC.squeeze(-1);
    C.resize(shapeC);

    auto px = A.data(), py = B.data();
    auto pz = C.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, batch, 16), [=](auto r) {
        for (int i = r.begin(); i < r.end(); i++) {
            impl::gemm(m, n, k, px + i*off_a, lda, py + i*off_b, ldb, pz + i*off_c, ldc);
        }
    });
}

template <typename T>
void matmul(const DevTensor<T>& A, const DevTensor<T>& B, DevTensor<T>& C) {
    if (A.rank() <= 2 && B.rank() <= 2) {
        dot(A, B, &C);
        return;
    }

    Shape shapeA = A.shape();
    Shape shapeB = B.shape();
    Shape shapeC;
    int batch;

    batch = detail::matmul_broadcast(shapeA, shapeB, shapeC);

    int m = shapeA.extent(shapeA.rank() - 2);
    int k = shapeA.extent(shapeA.rank() - 1);
    int n = shapeB.extent(shapeB.rank() - 1);
    int lda = std::max<int>(shapeA.stride(shapeA.rank() - 2), k);
    int ldb = std::max<int>(shapeB.stride(shapeB.rank() - 2), n);
    int ldc = std::max<int>(shapeC.stride(shapeC.rank() - 2), n);
    int off_a = shapeA.stride(shapeA.rank() - 3);
    int off_b = shapeB.stride(shapeB.rank() - 3);
    int off_c = shapeC.stride(shapeC.rank() - 3);

    if (A.rank() == 1)
        shapeC = shapeC.squeeze(-2);
    if (B.rank() == 1)
        shapeC = shapeC.squeeze(-1);
    C.resize(shapeC);

    std::vector<size_t> a_offsets(batch);
    std::vector<size_t> b_offsets(batch);
    std::vector<size_t> c_offsets(batch);
    std::vector<T> alpha(batch), beta(batch);

    for (int i = 0; i < batch; i++) {
        a_offsets[i] = i * off_a;
        b_offsets[i] = i * off_b;
        c_offsets[i] = i * off_c;
        alpha[i] = T{1};
        beta[i] = T{0};
    }

    gpgpu::blas::gemmBatched(
        gpgpu::blas::Layout::RowMajor,
        gpgpu::blas::Transpose::NoTrans,
        gpgpu::blas::Transpose::NoTrans,
        m, n, k,
        &alpha[0],
        A.data(), &a_offsets[0], lda,
        B.data(), &b_offsets[0], ldb,
        &beta[0],
        C.data(), &c_offsets[0], ldc,
        batch);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT> matmul(const TensorT& A, const TensorT& B) {
    if (A.rank() <= 2 && B.rank() <= 2) {
        return dot(A, B);
    } else {
        TensorT C;
        matmul(A, B, C);
        return C;
    }
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT> matpow(TensorT&& A, long n) {
    assert(A.is_square() && n >= 0);
    if (n == 0)
        return Tensor<tensor_value_type<TensorT>>::identity(2, A.extent(0));
    if (n == 1)
        return std::forward<TensorT>(A);
    n--;

    auto x = std::forward<TensorT>(A);
    auto y = x, t = x;
    while (n > 0) {
        if (n & 1)
            std::swap(y, dot(x, y, &t));
        std::swap(x, dot(x, x, &t));
        n >>= 1;
    }
    return y;
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
// Tensor shape operations
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
enable_if_tensor<TensorT> flatten(TensorT&& tensor, int axis) {
    tensor_type<TensorT> ret = std::forward<TensorT>(tensor);
    ret.flatten(axis);
    return ret;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT> squeeze(TensorT&& tensor, const std::vector<int>& axes = {}) {
    tensor_type<TensorT> ret = std::forward<TensorT>(tensor);
    ret.squeeze(axes);
    return ret;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT> unsqueeze(TensorT&& tensor, const std::vector<int>& axes) {
    tensor_type<TensorT> ret = std::forward<TensorT>(tensor);
    ret.unsqueeze(axes);
    return ret;
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

template <typename TensorT>
enable_if_tensor<TensorT, void>
slice(const TensorT& X, TensorT& Y,
      const std::vector<int>& starts, const std::vector<int>& ends,
      const std::vector<int>& axes, const std::vector<int>& steps)
{
    Shape slice_shape = X.shape().slice(starts, ends, axes, steps);
    reorder(X, slice_shape, Y.resize(slice_shape));
}

template <typename TensorT>
enable_if_tensor<TensorT, void>
slice(const TensorT& X, TensorT& Y, const std::vector<SliceDim>& dims) {
    Shape slice_shape = X.shape().slice(dims);
    reorder(X, slice_shape, Y.resize(slice_shape));
}

template <typename TensorT>
inline enable_if_tensor<TensorT, void> broadcast(const TensorT& src, TensorT& dst) {
    reorder(src, src.shape().broadcast(dst.shape()), dst);
}

template <typename TensorT>
std::enable_if_t<std::is_same<typename tensor_traits<TensorT>::tag, detail::cpu<void>>::value>
inline where(const tensor_type<TensorT, bool>& C, const TensorT& X, const TensorT& Y, tensor_type<TensorT>& Z) {
    auto z_shape = Shape::broadcast(C, X, Y);
    Z.resize(z_shape);

    auto c_shape = C.shape().broadcast(z_shape);
    auto x_shape = X.shape().broadcast(z_shape);
    auto y_shape = Y.shape().broadcast(z_shape);
    using T = tensor_value_type<TensorT>;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, Z.size(), GRAINSIZE), [&](auto r) {
        auto c_it = const_shaped_iterator<bool>(c_shape, C.data(), r.begin());
        auto x_it = const_shaped_iterator<T>(x_shape, X.data(), r.begin());
        auto y_it = const_shaped_iterator<T>(y_shape, Y.data(), r.begin());
        auto pz = Z.data() + r.begin();
        for (int count = r.size(); --count >= 0; ++pz, ++c_it, ++x_it, ++y_it) {
            *pz = *c_it ? *x_it : *y_it;
        }
    });
}

template <typename TensorT>
std::enable_if_t<
    is_tensor<TensorT>::value &&
    !std::is_same<typename tensor_traits<TensorT>::tag, detail::cpu<void>>::value>
inline where(const tensor_type<TensorT, bool>& C, const TensorT& X, const TensorT& Y, tensor_type<TensorT>& Z) {
    auto final_shape = Shape::broadcast(C, X, Y);
    Z.resize(final_shape);

    auto c_shape = C.shape().broadcast(final_shape);
    auto x_shape = X.shape().broadcast(final_shape);
    auto y_shape = Y.shape().broadcast(final_shape);

    gpgpu::dnn::where(
        final_shape.size(), final_shape.rank(),
        C.data(), c_shape.offset(), c_shape.extents(), c_shape.strides(),
        X.data(), x_shape.offset(), x_shape.extents(), x_shape.strides(),
        Y.data(), y_shape.offset(), y_shape.extents(), y_shape.strides(),
        Z.data(), 0);
}

template <typename TensorT>
enable_if_tensor<TensorT> where(const tensor_type<TensorT,bool>& C, const TensorT& X, const TensorT& Y) {
    tensor_type<TensorT> Z;
    where(C, X, Y, Z);
    return Z;
}

} // namespace dlf
