#pragma once

#include "./linalg_detail.h"

namespace dlf {

//==-------------------------------------------------------------------------
// Low level BLAS routines
//==-------------------------------------------------------------------------

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
gemv(cblas::Transpose trans,
     const tensor_value_type<TensorT>& alpha,
     const TensorT& A, const TensorT& x,
     const tensor_value_type<TensorT>& beta,
     TensorT& y)
{
    assert(A.is_matrix() && x.is_vector());
    auto m = A.extent(0), n = A.extent(1);
    if (trans == cblas::Transpose::NoTrans) {
        if (n != x.extent(0))
            throw shape_error("gemv: incompatible shape");
        y.resize(m);
    } else {
        if (m != x.extent(0))
            throw shape_error("gemv: incompatible shape");
        y.resize(n);
    }
    detail::gemv(trans, m, n, alpha, A.data(), A.stride(0), x.data(), 1, beta, y.data(), 1);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, tensor_type<TensorT>>
gemv(const TensorT& A, const TensorT& x) {
    using T = tensor_value_type<TensorT>;
    tensor_type<TensorT> y{};
    gemv(cblas::Transpose::NoTrans, xfn::one<T>(), A, x, xfn::zero<T>(), y);
    return y;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
gemm(cblas::Transpose transA, cblas::Transpose transB,
     const tensor_value_type<TensorT>& alpha,
     const TensorT& A, const TensorT& B,
     const tensor_value_type<TensorT>& beta,
     TensorT* C)
{
    assert(A.is_matrix() && B.is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA != cblas::Transpose::NoTrans)
        std::swap(m, k);
    if (transB != cblas::Transpose::NoTrans)
        std::swap(p, n);
    if (k != p)
        throw shape_error("gemm: incompatible shape");
    C->resize({m, n});

    detail::gemm(
        transA, transB,
        m, n, k,
        alpha,
        A.data(), A.stride(0),
        B.data(), B.stride(0),
        beta,
        C->data(), C->stride(0));
}

template <typename TensorT>
inline enable_if_non_view_tensor<TensorT, void>
gemm(cblas::Transpose transA, cblas::Transpose transB,
     const tensor_value_type<TensorT>& alpha, const TensorT& A, const TensorT& B,
     const tensor_value_type<TensorT>& beta, const TensorT& C, TensorT& Y)
{
    broadcast(C, Y);
    gemm(transA, transB, alpha, A, B, beta, &Y);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
gemm(cblas::Transpose transA, cblas::Transpose transB,
     const tensor_value_type<TensorT>& alpha, const TensorT& A, const TensorT& B,
     const tensor_value_type<TensorT>& beta, const TensorT& C)
{
    assert(A.is_matrix() && B.is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA != cblas::Transpose::NoTrans)
        std::swap(m, k);
    if (transB != cblas::Transpose::NoTrans)
        std::swap(p, n);
    assert(k == p);

    auto Y = C.broadcast_to({m, n}).copy();
    gemm(transA, transB, alpha, A, B, beta, &Y);
    return Y;
}

//==-------------------------------------------------------------------------
// symmetric matrix multiplication
//==-------------------------------------------------------------------------

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
symv(cblas::Triangle uplo,
     const tensor_value_type<TensorT>& alpha,
     const TensorT& A, const TensorT& x,
     const tensor_value_type<TensorT>& beta,
     TensorT& y)
{
    assert(A.is_matrix() && x.is_vector());
    auto n = x.extent(0);
    if (A.extent(0) < n || A.extent(1) < n)
        throw shape_error("symv: matrix A has too few dimensions");
    y.resize(n);
    detail::symv(uplo, n, alpha, A.data(), A.stride(0), x.data(), 1, beta, y.data(), 1);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, tensor_type<TensorT>>
symv(cblas::Triangle uplo, const TensorT& A, const TensorT& x) {
    using T = tensor_value_type<TensorT>;
    tensor_type<TensorT> y{};
    symv(uplo, xfn::one<T>(), A, x, xfn::zero<T>(), y);
    return y;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
symm(cblas::Side side, cblas::Triangle uplo,
     const tensor_value_type<TensorT>& alpha,
     const TensorT& A, const TensorT& B,
     const tensor_value_type<TensorT>& beta,
     TensorT& C)
{
    assert(A.is_matrix() && B.is_matrix());

    auto m = B.extent(0), n = B.extent(1);
    auto k = (side == cblas::Side::Left) ? m : n;
    if (A.extent(0) < k || A.extent(1) < k)
        throw shape_error("symm: matrix A has too few dimensions");
    C.resize(m, n);

    detail::symm(side, uplo, m, n,
                 alpha,
                 A.data(), A.stride(0),
                 B.data(), B.stride(0),
                 beta,
                 C.data(), C.stride(0));
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, tensor_type<TensorT>>
symm(cblas::Side side, cblas::Triangle uplo, const TensorT& A, const TensorT& B) {
    using T = tensor_value_type<TensorT>;
    tensor_type<TensorT> C{};
    symm(side, uplo, xfn::one<T>(), A, B, xfn::zero<T>(), C);
    return C;
}

//==-------------------------------------------------------------------------
// Triangular matrix multiplication
//==-------------------------------------------------------------------------

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
trmv(cblas::Triangle uplo, cblas::Transpose trans, cblas::Diagonal diag,
     const TensorT& A, const TensorT& x, TensorT& y)
{
    assert(A.is_matrix() && x.is_vector());
    auto n = x.extent(0);
    if (A.extent(0) < n || A.extent(1) < n)
        throw shape_error("trmv: matrix A has too few dimensions");
    reorder(x, y);
    detail::trmv(uplo, trans, diag, n, A.data(), A.stride(0), y.data(), 1);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, tensor_type<TensorT>>
trmv(cblas::Triangle uplo, cblas::Transpose trans, cblas::Diagonal diag,
     const TensorT& A, const TensorT& x)
{
    tensor_type<TensorT> y{};
    trmv(uplo, trans, diag, A, x, y);
    return y;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
trmm(cblas::Side side, cblas::Triangle uplo, cblas::Transpose transA, cblas::Diagonal diag,
     const tensor_value_type<TensorT>& alpha,
     const TensorT& A, const TensorT& B, TensorT& C)
{
    assert(A.is_matrix() && B.is_matrix());

    auto m = B.extent(0), n = B.extent(1);
    auto k = (side == cblas::Side::Left) ? m : n;
    if (A.extent(0) < k || A.extent(1) < k)
        throw shape_error("trmm: matrix A has too few dimensions");
    C.resize(m, n);
    flat_copy(B, C);

    detail::trmm(side, uplo, transA, diag,
                 m, n, alpha,
                 A.data(), A.stride(0),
                 C.data(), C.stride(0));
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, tensor_type<TensorT>>
trmm(cblas::Side side, cblas::Triangle uplo, cblas::Transpose transA, cblas::Diagonal diag,
     const TensorT& A, const TensorT& B)
{
    using T = tensor_value_type<TensorT>;
    tensor_type<TensorT> C{};
    trmm(side, uplo, transA, diag, xfn::one<T>(), A, B, C);
    return C;
}

//==-------------------------------------------------------------------------
// Extended matrix multiplication
//==-------------------------------------------------------------------------

/**
 * Matrix product of two tensors.
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
template <typename LHS, typename RHS, typename RET>
std::enable_if_t<
    is_exactly_same_tensor<LHS, RET>::value &&
    is_exactly_same_tensor<RHS, RET>::value &&
    !std::is_const<std::remove_reference_t<RET>>::value>
matmul(const tensor_value_type<LHS>& alpha,
       const LHS& A, const RHS& B,
       const tensor_value_type<LHS>& beta,
       RET&& C)
{
    Shape shapeA = A.shape();
    Shape shapeB = B.shape();
    Shape shapeC;

    // broadcast and normalize on input shapes
    int batch_size = detail::matmul_broadcast(shapeA, shapeB, shapeC);

    int m = shapeA.extent(-2);
    int k = shapeA.extent(-1);
    int n = shapeB.extent(-1);

    int lda = shapeA.stride(-2);
    int ldb = shapeB.stride(-2);
    int incA = shapeA.stride(-1);
    int incB = shapeB.stride(-1);

    if (lda == 0 && m == 1)
        lda = k;
    if (ldb == 0 && k == 1)
        ldb = n;
    if (incA == 0 && k == 1)
        incA = 1;
    if (incB == 0 && n == 1)
        incB = 1;

    // blas cannot do gemm on non-contiguous shape, reorder if necessary
    auto dataA = A.data();
    auto dataB = B.data();
    tensor_type<LHS> tmpA{};
    tensor_type<RHS> tmpB{};

    bool reorderA = false, reorderB = false;
    detail::is_matmul_reorder_needed<LHS, RHS>(&reorderA, &reorderB, m, n, k, lda, incA, ldb, incB);
    if (reorderA) {
        reorder(A, tmpA);
        shapeA = tmpA.shape();
        dataA = tmpA.data();
        lda = k;
        incA = 1;
    }
    if (reorderB) {
        reorder(B, tmpB);
        shapeB = tmpB.shape();
        dataB = tmpB.data();
        ldb = n;
        incB = 1;
    }
    if (reorderA || reorderB) {
        batch_size = detail::matmul_broadcast(shapeA, shapeB, shapeC);
    }

    // remove 1 from final shape if one of input tensors is a vector
    if (A.rank() == 1)
        shapeC = shapeC.squeeze(-2);
    if (B.rank() == 1)
        shapeC = shapeC.squeeze(-1);

    // get actual shape of C
    C.resize(shapeC);
    shapeC = C.shape();
    if (B.rank() == 1)
        shapeC = shapeC.unsqueeze(-1);
    if (A.rank() == 1)
        shapeC = shapeC.unsqueeze(-2);

    int ldc = shapeC.stride(-2);
    int incC = shapeC.stride(-1);
    if (ldc == 0 && m == 1)
        ldc = n;
    if (incC == 0 && n == 1)
        incC = 1;
    if (incC != 1 || ldc < n) // BLAS requires matrix C contiguous on column
        throw shape_error("matmul: the output tensor must be contiguous");

    // do batched matrix multiplication
    detail::matmul(
        m, n, k,
        alpha,
        shapeA, dataA, lda, incA,
        shapeB, dataB, ldb, incB,
        beta,
        shapeC, C.data(), ldc,
        batch_size);
}

template <typename LHS, typename RHS, typename RET>
std::enable_if_t<
    is_exactly_same_tensor<LHS, RET>::value &&
    is_exactly_same_tensor<RHS, RET>::value &&
    !std::is_const<std::remove_reference_t<RET>>::value>
matmul(const LHS& A, const RHS& B, RET&& C) {
    using T = tensor_value_type<LHS>;
    matmul(xfn::one<T>(), A, B, xfn::zero<T>(), C);
}

template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
inline matmul(const LHS& A, const RHS& B) {
    tensor_type<LHS> C{};
    matmul(A, B, C);
    return C;
}

//==-------------------------------------------------------------------------

/**
 * Matrix exponentiation.
 */
template <typename TensorT>
enable_if_tensor<TensorT> matpow(TensorT&& A, long n) {
    if (!A.is_inner_square())
        throw shape_error("matpow: square matrix required");
    if (n < 0)
        throw std::logic_error("matpow: negative exponentiation is not supported");

    if (n == 0)
        return tensor_type<TensorT>::identity(A.shape());
    if (n == 1)
        return std::forward<TensorT>(A);
    n--;

    tensor_type<TensorT> x = std::forward<TensorT>(A);
    auto y = x, t = x;
    while (n > 0) {
        if (n & 1) {
            matmul(x, y, t);
            std::swap(y, t);
        }
        matmul(x, x, t);
        std::swap(x, t);
        n >>= 1;
    }
    return y;
}

//==-------------------------------------------------------------------------

/**
 * Dot product of two tensors. Specifically,
 *
 *  - If both A and B are 1-D tensors, it is inner product of vectors (without
 *    complex conjugation).
 *
 *  - If both A and B are 2-D tensors, it is matrix multiplication, but using
 *    matmul is preferred.
 *
 *  - If either A and B is 0-D (scalar), it is equivalent to multiply and using
 *    A * B is preferred.
 *
 *  - If A is an N-D tensor and B is a 1-D tensor, it is a sum product over the
 *    last axis of A and B.
 *
 *  - If A is an N-D tensor and B is a M-D tensor (where M>=2), it is sum product
 *    over the last axis of A and the second-to-last axis of B:
 *
 *    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
 */
template <typename LHS, typename RHS, typename RET>
std::enable_if_t<
    is_exactly_same_tensor<LHS, RET>::value &&
    is_exactly_same_tensor<RHS, RET>::value &&
    !std::is_const<std::remove_reference_t<RET>>::value>
dot(const LHS& A, const RHS& B, RET&& C) {
    if (A.is_scalar() || B.is_scalar()) {
        transformTo(A, B, C, xfn::multiplies<>());
        return;
    }

    if ((A.rank() <= 2 && B.rank() <= 2) || B.rank() == 1) {
        matmul(A, B, C);
        return;
    }

    std::vector<size_t> c_dims;
    for (int i = 0; i < A.rank()-1; ++i)
        c_dims.push_back(A.extent(i));
    for (int i = 0; i < B.rank()-2; ++i)
        c_dims.push_back(B.extent(i));
    c_dims.push_back(B.extent(-1));

    std::vector<int> a_unsq;
    for (int i = 0; i < B.rank()-1; i++) {
        a_unsq.push_back(i + A.rank() - 1);
    }

    C.resize(Shape(c_dims));
    matmul(unsqueeze(A, a_unsq), B, unsqueeze(C, -2));
}

template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
dot(const LHS& A, const RHS& B) {
    tensor_type<LHS> C{};
    dot(A, B, C);
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
template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
inline operator,(const LHS& lhs, const RHS& rhs) {
    return dot(lhs, rhs);
}
} // namespace dot_product

//==-------------------------------------------------------------------------

/**
 * Compute the dot product of two or more tensors in a single function call,
 * while automatically selecting the fastest evaluation order.
 *
 * multi_dot chains dot and uses optimal parenthesization of the matrices,
 * this can speed up the multiplication a lot.
 *
 * If the first argument is 1-D it is treated as a row vector, If the last
 * argument is 1-D it is treated as a column vector. The other arguments
 * must be 2-D.
 *
 * @see https://en.wikipedia.org/wiki/Matrix_chain_multiplication
 */
template <typename TensorT>
std::enable_if_t<is_tensor_view<TensorT>::value, tensor_type<TensorT>>
multi_dot(const std::vector<TensorT>& args) {
    if (args.size() == 0)
        throw std::logic_error("multi_dot: at least 1 argument is required");
    if (args.size() == 1)
        return args[0];

    auto dims = detail::get_matrix_chain_dimensions(args);
    auto s = detail::matrix_chain_order(dims);
    return detail::optimal_parenthesizations(s, args);
}

template <typename First, typename... Rest>
std::enable_if_t<
    is_tensor<First>::value &&
    cxx::conjunction<is_same_tensor<First, Rest>...>::value,
    tensor_type<First>>
inline multi_dot(const First& first, const Rest&... rest) {
    return multi_dot(std::vector<tensor_view_type<First>>{first.view(), rest.view()...});
}

//==-------------------------------------------------------------------------

/**
 * Compute tensor dot product along specified axes.
 *
 * Given two tensors, A and B, and two axes array, a_axes and b_axes, sum the
 * products of A's and B's elements (components) over the axes specified by
 * a_axes and b_axes.
 *
 * @param A, B Tensors to "dot".
 * @param axes_a, axes_b
 *        A list of axes to be summed over. Both elements must be of the same
 *        length.
 */
template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
tensordot(const LHS& A, const RHS& B, std::vector<int> axes_a, std::vector<int> axes_b) {
    bool equal = false;
    if (axes_a.size() == axes_b.size()) {
        equal = true;
        for (int k = 0; k < axes_a.size(); k++) {
            detail::norm_axis(A.rank(), axes_a[k]);
            detail::norm_axis(B.rank(), axes_b[k]);
            if (A.extent(axes_a[k]) != B.extent(axes_b[k])) {
                equal = false;
                break;
            }
        }
    }
    if (!equal) {
        throw shape_error("tensordot: shape-mismatch");
    }

    std::vector<int> out_dims;

    // Move the axes to sum over to the end of A
    std::vector<size_t> newaxes_a;
    int M = 1, K = 1;
    for (int k = 0; k < A.rank(); k++) {
        if (std::find(axes_a.begin(), axes_a.end(), k) == axes_a.end()) {
            newaxes_a.push_back(k);
            out_dims.push_back(A.extent(k));
            M *= A.extent(k);
        }
    }
    for (auto k : axes_a) {
        newaxes_a.push_back(k);
        K *= A.extent(k);
    }

    // Move the axes to sum over to the front of B
    std::vector<size_t> newaxes_b;
    int P = 1, N = 1;
    for (auto k : axes_b) {
        newaxes_b.push_back(k);
        P *= B.extent(k);
    }
    for (int k = 0; k < B.rank(); k++) {
        if (std::find(axes_b.begin(), axes_b.end(), k) == axes_b.end()) {
            newaxes_b.push_back(k);
            out_dims.push_back(B.extent(k));
            N *= B.extent(k);
        }
    }

    auto at = reshape(A.transpose(newaxes_a), {M, K});
    auto bt = reshape(B.transpose(newaxes_b), {P, N});
    auto res = matmul(at, bt);
    res.reshape(out_dims);
    return res;
}

/**
 * Compute tensor dot product along specified axis
 *
 * Given two tensors, A and B, and an axis N, the products of last N dimensions
 * of A and the first N dimensions of B are summed over.
 *
 * Three common use cases are:
 *
 *     - N = 0: tensor product
 *     - N = 1: tensor dot product
 *     - N = 2: (default) tensor double contraction
 *
 * The sequence for evaluation will be: first the -Nth axis in A and 0th axis in
 * B, and the -1th axis in A and Nth axis in B last.
 *
 * @param A, B Tensors to "dot".
 * @param N Sum over the last N axes of A and the first N axes of B in order.
 *        The size of the corresponding axes must match.
 */
template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
tensordot(const LHS& A, const RHS& B, int N = 2) {
    std::vector<int> axes_a, axes_b;
    for (int i = 0; i < N; i++) {
        axes_a.push_back(i-N);
        axes_b.push_back(i);
    }
    return tensordot(A, B, axes_a, axes_b);
}

//==-------------------------------------------------------------------------

/**
 * Inner product of two tensors.
 *
 * Ordinary inner product of vectors for 1-D tensor (without complex conjugation),
 * in higher dimensions a sum product over the last axes.
 *
 * For vectors (1-D tensor) it computes the ordinary inner-product.
 *
 * More generally, if A.rank() > 0 and B.rank() > 0:
 *
 *     inner(A, B) = tensordot(A, B, {-1}, {-1})
 *
 * In addition a or b may be scalars, in which case:
 *
 *     inner(A, B) = A * B
 */
template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
inner(const LHS& A, const RHS& B) {
    if (A.is_scalar() || B.is_scalar())
        return A * B;
    if (A.is_vector() && B.is_vector())
        return dot(A, B);
    return tensordot(A, B, {-1}, {-1});
}

//==-------------------------------------------------------------------------

/**
 * The outer product on tensors is typically referred to as the tensor product.
 * Given a tensor a of order q with dimensions (i1, ..., iq), and a tensor b
 * of order r with dimensions (j1, ..., jr), their outer product c is of order
 * q + r with dimensions (k1, ..., kq+r) which are the i dimensions followed
 * by the j dimensions.
 */
template <typename LHS, typename RHS, typename Fn>
enable_if_tensors<LHS, RHS, Fn>
inline outer(const LHS& A, const RHS& B, Fn f) {
    auto rank = A.rank() + B.rank();
    return transform(unsqueeze_right(A, rank), unsqueeze_left(B, rank), f);
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::multiplies<>>
inline outer(const LHS& A, const RHS& B) {
    return outer(A, B, xfn::multiplies<>());
}

//==-------------------------------------------------------------------------

/**
 * Computes the Kronecker product, a composite tensor made of blocks of the second
 * tensor scaled by the first.
 *
 * The function assumes that the number of dimensions of a and b are the same,
 * if necessary prepending the smallest with ones. If A.shape = (r0,r1,...,rN) and
 * B.shape = (s0,s1,...,sN), the kronecker product has shape (r0*s0,r1*s1,...,rN*sN).
 * The elements are products of elements from A and B, organized explicitly by:
 *
 *     kron(A,B)[k0,k1,...,kN] = A[i0,i1,...,iN] * B[j0,j1,...,jN]
 *
 * where:
 *
 *     kt = it * st + jt,  t = 0,...,N
 */
template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::multiplies<>>
kronecker(const LHS& A, const RHS& B) {
    auto rank = std::max(A.rank(), B.rank());
    auto a_dims = A.shape().extents();
    auto b_dims = B.shape().extents();
    auto c_dims = std::vector<size_t>(rank);

    while (a_dims.size() < rank)
        a_dims.insert(a_dims.begin(), 1);
    while (b_dims.size() < rank)
        b_dims.insert(b_dims.begin(), 1);
    for (int i = 0; i < rank; i++)
        c_dims[i] = a_dims[i] * b_dims[i];

    tensor_invoke_result<xfn::multiplies<>, LHS, RHS> C(Shape{c_dims});
    transformTo(unsqueeze_right(unsqueeze_left(A, rank), rank*2),
                unsqueeze_left(B, rank*2),
                partition(C, b_dims),
                xfn::multiplies<>());
    return C;
}

//==-------------------------------------------------------------------------

/**
 * Return a batched diagonal tensor with a given batched diagonal values.
 *
 * Given a diagonal, this operation returns a tensor with the diagonal and
 * everything else padded with zeros. The diagonal is computed as follows:
 *
 * Assume diagonal has k dimensions [I, J, K, ..., N], then the output is
 * a tensor of rank k+1 with dimensions [I, J, K, ..., N, N] where:
 *
 * output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n].
 */
template <typename TensorT>
enable_if_tensor<TensorT> diag(const TensorT& diagonal) {
    assert(diagonal.rank() > 0);
    auto dims = diagonal.shape().extents();
    dims.push_back(dims[dims.size()-1]);

    tensor_type<TensorT> result(Shape{dims});
    result.fill(0);
    reorder(diagonal, result.diagonal());
    return result;
}

/**
 * Return the sum along diagonals of the array.
 *
 * If X is 2-D, the sum along its diagonal with the given offset is returned.
 *
 * If X has more than two dimensions, then the axes specified by axis1 and
 * axis2 are used to determine the 2-D sub-matrices whose traces are returned.
 * The shape of the resulting array is the same as that of with axis1 and axis2
 * removed.
 *
 * @param X Input tensor, from which the diagonals are taken.
 * @param offset Offset of the diagonal from the main diagonal. Can be both
 *        positive and negative. Defaults to 0.
 * @param axis1, axis2
 *        Axes to be used as the first and second axis of the 2-D sub-matrices
 *        from which the diagonals should be taken. Defaults are the last two
 *        axes of X.
 */
template <typename TensorT>
enable_if_tensor<TensorT>
trace(const TensorT& X, int offset = 0, int axis1 = -2, int axis2 = -1) {
    return reduce_sum(X.diagonal(offset, axis1, axis2), {-1}, false);
}

/**
 * Matrix or vector norm.
 *
 * This function is able to return one of eight different matrix norms, or one
 * of an infinite number of vector norms (described below), depending on the
 * value of the `ord` parameter.
 *
 * @param X Input tensor. If `axis` is omitted, `X` must be 1-D or 2D.
 * @param ord Order of the norm (see table under Notes).
 * @param axis Specifies the axis of `X` along which to compute the norms.
 * @param keepdims If this is set to true, the axes which are normed over are
 *        left in the result as dimensions with size one.
 *
 * @note
 *
 * The following norms can be calculated:
 *
 * =====  ============================  =======================
 * ord    norm for matrices             norm for vectors
 * =====  ============================  =======================
 * inf    max(sum(abs(x), axis=1)       max(abs(x))
 * 1      max(sum(abs(x), axis=0)       as below
 * 2      2-norm (largest sing. value)  as below
 * other  --                            sum(abs(x)^ord)^(1/ord)
 * =====  ============================  =======================
 */
template <typename TensorT>
enable_if_tensor<TensorT>
norm(const TensorT& X, float ord, int axis, bool keepdims = false) {
    if (ord < 1.f)
        throw std::logic_error("norm: invalid norm order for vector");
    if (std::isinf(ord))
        return reduce_amax(X, axis, keepdims);
    if (ord == 1.f) // special case for speed up
        return reduce_asum(X, axis, keepdims);
    if (std::isnan(ord) || ord == 2.f) // special case for speed up
        return reduce_nrm2(X, axis, keepdims);
    return detail::norm_p(X, ord, axis, keepdims);
}

template <typename TensorT>
enable_if_tensor<TensorT>
norm(const TensorT& X, float ord, int row_axis, int col_axis, bool keepdims = false) {
    detail::norm_axes(X.rank(), row_axis, col_axis);

    if (std::isnan(ord)) {
        // Frobenius norm
        return reduce_nrm2(X, {row_axis, col_axis}, keepdims);
    }

    if (ord > 0.f && std::isinf(ord)) {
        if (row_axis > col_axis && !keepdims)
            row_axis--;
        return reduce_max(reduce_asum(X, col_axis, keepdims), row_axis, keepdims);
    }

    if (ord == 1.f) {
        if (col_axis > row_axis && !keepdims)
            col_axis--;
        return reduce_max(reduce_asum(X, row_axis, keepdims), col_axis, keepdims);
    }

    // TODO: for ord == 2, we need singular value decomposition
    throw std::logic_error("norm: invalid norm order for matrices");
}

template <typename TensorT>
enable_if_tensor<TensorT>
norm(const TensorT& X, float ord, bool keepdims = false) {
    if (X.rank() == 1)
        return norm(X, ord, 0, keepdims);
    if (X.rank() == 2)
        return norm(X, ord, 0, 1, keepdims);
    throw shape_error("norm: improper number of dimensions to norm");
}

/**
 * Compute the inverse of the given matrix.
 */
template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value, Tensor<T>>
matinv(Tensor<T>&& X) {
    if (!X.is_inner_square())
        throw shape_error("matinv: requires square matrix");

    lapack_int n  = X.extent(-1);
    lapack_int m  = X.size() / (n*n);
    lapack_int ld = X.stride(-2);
    std::vector<lapack_int> ipiv(n);

    auto px = X.data();
    for (lapack_int i = 0; i < m; ++i, px += n*n) {
        if (cblas::getrf(n, n, px, ld, ipiv.data()) != 0)
            throw std::runtime_error("matinv: the matrix is not invertible");
        if (cblas::getri(n, px, ld, ipiv.data()) != 0)
            throw std::runtime_error("matinv: the matrix is not invertible");
    }
    return std::move(X);
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value, Tensor<T>>
inline matinv(const Tensor<T>& X) {
    return matinv(Tensor<T>(X));
}

/**
 * Compute the determinant of the given matrix.
 */
template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value, Tensor<T>>
det(Tensor<T>&& X) {
    if (!X.is_inner_square())
        throw shape_error("det: requires square matrix");

    lapack_int n  = X.extent(-1);
    lapack_int ld = X.stride(-2);
    std::vector<lapack_int> ipiv(n);

    auto y_dims = X.shape().extents();
    y_dims.erase(y_dims.end()-2, y_dims.end());
    auto Y = Tensor<T>(Shape(y_dims));

    auto px = X.data();
    for (auto& y : Y) {
        cblas::getrf(n, n, px, ld, ipiv.data());
        y = T{1};
        for (lapack_int i = 0; i < n; ++i) {
            y *= px[i*(n+1)];
            if (i+1 != ipiv[i])
                y = -y;
        }
        px += n*n;
    }
    return Y;
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value, Tensor<T>>
inline det(const Tensor<T>& X) {
    return det(Tensor<T>(X));
}

template <typename T>
class linear_solve_function {
    Tensor<T> A;
    std::vector<lapack_int> ipiv;
    lapack_int info;

public:
    linear_solve_function(Tensor<T>&& A);
    linear_solve_function(const Tensor<T>& A);

    bool has_solution() const { return info == 0; }

    Tensor<T> operator()(Tensor<T>&& b) const;
    Tensor<T> operator()(const Tensor<T>& b) const;
};

template <typename T>
linear_solve_function<T>::linear_solve_function(Tensor<T>&& mat) : A(std::move(mat)) {
    if (!A.is_square())
        throw shape_error("solve: requires square matrix");

    auto n = A.extent(-1);
    ipiv.resize(n);
    info = cblas::getrf(n, n, A.data(), A.stride(0), ipiv.data());
}

template <typename T>
linear_solve_function<T>::linear_solve_function(const Tensor<T>& A)
    : linear_solve_function(Tensor<T>(A)) {}

template <typename T>
Tensor<T> linear_solve_function<T>::operator()(Tensor<T>&& b) const {
    if (info != 0)
        throw std::runtime_error("solve: the linear equation has no solution");
    if (!(b.is_vector() || b.is_matrix()))
        throw shape_error("solve: the rhs must be a vector or a matrix");

    lapack_int n = A.extent(0);
    lapack_int nrhs = b.is_vector() ? 1 : b.extent(1);
    if (b.extent(0) != n)
        throw shape_error("solve: incompatible shape");

#if HAS_LAPACKE
    cblas::getrs(n, nrhs, A.data(), A.stride(0), ipiv.data(), b.data(), b.stride(0));
    return std::move(b);
#else
    if (nrhs == 1) {
        cblas::getrs(n, 1, A.data(), n, ipiv.data(), b.data(), n);
        return std::move(b);
    } else {
        auto x = b.transpose().reorder();
        cblas::getrs(n, nrhs, A.data(), n, ipiv.data(), x.data(), n);
        reorder(x.transpose(), b);
        return std::move(b);
    }
#endif
}

template <typename T>
Tensor<T> linear_solve_function<T>::operator()(const Tensor<T>& b) const {
    return operator()(Tensor<T>(b));
}

/**
 * Generate a linear solve function that can be applied repeatedly to different b.
 */
template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value, linear_solve_function<T>>
inline solve(Tensor<T>&& A) {
    return linear_solve_function<T>(std::move(A));
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value, linear_solve_function<T>>
inline solve(const Tensor<T>& A) {
    return linear_solve_function<T>(A);
}

/**
 * Solve a linear equation Ax=b.
 */
template <typename TensorA, typename TensorB>
std::enable_if_t<
    is_cpu_tensor<TensorA>::value && is_cpu_tensor<TensorB>::value &&
    !is_tensor_view<TensorA>::value && !is_tensor_view<TensorB>::value &&
    is_exactly_same_tensor<TensorA, TensorB>::value &&
    cblas::is_blasable<tensor_value_type<TensorA>>::value,
    tensor_type<TensorB>>
inline solve(TensorA&& A, TensorB&& b) {
    return solve(std::forward<TensorA>(A))(std::forward<TensorB>(b));
}

} // namespace dlf
