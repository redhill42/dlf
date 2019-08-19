#pragma once

#include <unordered_set>

namespace dlf {

//==-------------------------------------------------------------------------
// Vector or matrix multiplication.
//==-------------------------------------------------------------------------

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
inline enable_if_non_view_tensor<TensorT, void>
gemm(const tensor_value_type<TensorT>& alpha, const TensorT& A, const TensorT& B,
     const tensor_value_type<TensorT>& beta, const TensorT& C, TensorT& Y,
     bool transA = false, bool transB = false, TensorT* work = nullptr)
{
    broadcast(C, Y);
    gemm(alpha, A, B, beta, &Y, transA, transB, work);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
gemm(const tensor_value_type<TensorT>& alpha, const TensorT& A, const TensorT& B,
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

} // namespace dlf
