#pragma once

#include <unordered_set>

namespace dlf {

//==-------------------------------------------------------------------------
// General matrix multiplication (Host)
//==-------------------------------------------------------------------------

template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
     const T& beta, Tensor<T>* C,
     bool transA = false, bool transB = false,
     Tensor<T>* = nullptr)
{
    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);
    const auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    if (k != p)
        throw shape_error("gemm: incompatible shape");
    C->resize({m, n});

    if (alpha == T(0)) {
        *C *= Tensor<T>::scalar(beta);
        return;
    }

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [&](auto&& r) {
        size_t incX = transA ? lda : 1;
        size_t incY = transB ? 1 : ldb;
        for (size_t i = r.rows().begin(); i != r.rows().end(); i++) {
            T* pz = &C->data()[i * ldc + r.cols().begin()];
            for (size_t j = r.cols().begin(); j != r.cols().end(); j++) {
                const T* px = A.data() + (transA ? i : i*lda);
                const T* py = B.data() + (transB ? j*ldb : j);
                T v = beta == T(0) ? T(0) : *pz * beta;
                for (size_t t = 0; t < k; t++) {
                    v += alpha * *px * *py;
                    px += incX;
                    py += incY;
                }
                *pz++ = std::move(v);
            }
        }
    });
}

template <typename T>
std::enable_if_t<cblas::RequireBlasType<T>>
gemm(const T& alpha, const Tensor<T>& A, const Tensor<T>& B,
     const T& beta, Tensor<T>* C,
     bool transA = false, bool transB = false,
     Tensor<T>* = nullptr)
{
    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    if (k != p)
        throw shape_error("gemm: incompatible shape");
    C->resize({m, n});

    cblas::gemm(cblas::Layout::RowMajor,
                transA ? cblas::Transpose::Trans : cblas::Transpose::NoTrans,
                transB ? cblas::Transpose::Trans : cblas::Transpose::NoTrans,
                m, n, k, alpha, A.data(), A.stride(0), B.data(), B.stride(0),
                beta, C->data(), C->stride(0));
}

template <typename T>
inline size_t gemmWorkspaceSize(
    const Tensor<T>&, const Tensor<T>&, const Tensor<T>&,
    bool = false, bool = false)
{
    return 0; // API compatible to DevTensor
}

//==-------------------------------------------------------------------------
// General matrix multiplication (Device)
//==-------------------------------------------------------------------------

template <typename T>
void gemm(const T& alpha, const DevTensor<T>& A, const DevTensor<T>& B,
          const T& beta, DevTensor<T>* C,
          bool transA = false, bool transB = false,
          DevTensor<T>* work = nullptr)
{
    assert(A.is_matrix() && B.is_matrix() && C->is_matrix());
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    if (k != p)
        throw shape_error("gemm: incompatible shape");
    C->resize({m, n});

    gblas::gemm(gblas::Layout::RowMajor,
                transA ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
                transB ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
                m, n, k,
                alpha,
                A.data(), A.stride(0),
                B.data(), B.stride(0),
                beta,
                C->data(), C->stride(0),
                work == nullptr ? nullptr : &work->data());
}

template <typename T>
inline size_t gemmWorkspaceSize(
    const DevTensor<T>& A, const DevTensor<T>& B, const DevTensor<T>& C,
    bool transA = false, bool transB = false)
{
    auto m = A.extent(0), k = A.extent(1);
    auto p = B.extent(0), n = B.extent(1);

    if (transA)
        std::swap(m, k);
    if (transB)
        std::swap(p, n);
    if (k != p || m != C.extent(0) || n != C.extent(1))
        throw shape_error("gemm: incompatible shape");

    return gblas::gemmTempBufferSize<T>(
        gblas::Layout::RowMajor,
        transA ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
        transB ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
        m, n, k, 0, A.stride(0), 0, B.stride(0), 0, C.stride(0));
}

//==-------------------------------------------------------------------------
// General matrix multiplication (Uniform)
//==-------------------------------------------------------------------------

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

//==-------------------------------------------------------------------------
// Extended matrix multiplication
//==-------------------------------------------------------------------------

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

    return prefixShape.rank() == 0 ? 1 : prefixShape.size();
}

template <int = 0>
int batch_offset(const Shape& shape, int batch) {
    int ret = shape.offset();
    for (int i = shape.rank() - 3; i >= 0; --i) {
        auto dim = shape.extent(i);
        auto index = batch % dim;
        batch /= dim;
        ret += index * shape.stride(i);
    }
    return ret;
}

template <int = 0>
bool is_contiguous_strides(const Shape& shape) {
    if (shape.rank() <= 2)
        return true;

    int stride = shape.stride(shape.rank() - 3);
    if (stride == 0) {
        for (int i = shape.rank() - 4; i >= 0; --i) {
            if (shape.stride(i) != 0)
                return false;
        }
    } else {
        for (int i = shape.rank() - 4; i >= 0; --i) {
            if (shape.stride(i) != stride*shape.extent(i))
                return false;
            stride = shape.stride(i);
        }
    }
    return true;
}

template <typename T>
std::enable_if_t<cblas::RequireBlasType<T>>
matmul_cpu(int m, int n, int k,
           const T* A, int lda, int incA,
           const T* B, int ldb, int incB,
           T* C, int ldc)
{
    if (m == 1 && n == 1) {
        *C = cblas::dot(k, A, incA, B, ldb);
        return;
    }

    auto transA = cblas::Transpose::NoTrans;
    if (incA != 1) {
        assert(lda == 1);
        transA = cblas::Transpose::Trans;
        lda = incA;
    }

    if (n == 1) {
        auto layout = transA == cblas::Transpose::NoTrans
            ? cblas::Layout::RowMajor
            : cblas::Layout::ColMajor;
        cblas::gemv(layout, cblas::Transpose::NoTrans,
                    m, k,
                    T{1},
                    A, lda, B, ldb,
                    T{0},
                    C, 1);
        return;
    }

    auto transB = cblas::Transpose::NoTrans;
    if (incB != 1) {
        assert(ldb == 1);
        transB = cblas::Transpose::Trans;
        ldb = incB;
    }

    cblas::gemm(cblas::Layout::RowMajor, transA, transB,
                m, n, k, T{1}, A, lda, B, ldb, T{0}, C, ldc);
}

template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
matmul_cpu(int m, int n, int k,
           const T* A, int lda, int incA,
           const T* B, int ldb, int incB,
           T* C, int ldc)
{
    if (m == 1 && n == 1) {
        *C = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, k, GRAINSIZE),
            T{},
            [=](auto r, T sum) {
                auto px = A + r.begin()*incA;
                auto py = B + r.begin()*ldb;
                for (size_t k = r.size(); k-- != 0; px += incA, py += ldb)
                    sum += *px * *py;
                return sum;
            },
            std::plus<T>());
    } else {
        tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [=](auto r) {
            for (size_t i = r.rows().begin(); i != r.rows().end(); i++) {
                for (size_t j = r.cols().begin(); j != r.cols().end(); j++) {
                    T v{};
                    for (size_t t = 0; t < k; t++)
                        v += A[i*lda + t*incA] * B[t*ldb + j*incB];
                    C[i*ldc + j] = std::move(v);
                }
            }
        });
    }
}

template <typename T>
void matmul(int m, int n, int k,
            const Shape& shapeA, const T* A, int lda, int incA,
            const Shape& shapeB, const T* B, int ldb, int incB,
            T* C, int ldc,
            int batch_size)
{
    if (batch_size == 1) {
        matmul_cpu(m, n, k, A+shapeA.offset(), lda, incA, B+shapeB.offset(), ldb, incB, C, ldc);
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0, batch_size, 16), [=](auto r) {
            for (int p = r.begin(); p < r.end(); p++) {
                matmul_cpu(
                    m, n, k,
                    A + batch_offset(shapeA, p), lda, incA,
                    B + batch_offset(shapeB, p), ldb, incB,
                    C + p*m*n, ldc);
            }
        });
    }
}

template <typename T>
void matmul(int m, int n, int k,
            const Shape& shapeA, const gpgpu::Buffer<T>& A, int lda, int incA,
            const Shape& shapeB, const gpgpu::Buffer<T>& B, int ldb, int incB,
            gpgpu::Buffer<T>& C, int ldc,
            int batch_size)
{
    if (batch_size == 1 && m == 1 && n == 1) {
        gblas::dot(
            k,
            A, shapeA.offset(), incA,
            B, shapeB.offset(), ldb,
            C, 0);
        return;
    }

    auto transA = gblas::Transpose::NoTrans;
    if (incA != 1) {
        assert(lda == 1);
        transA = gblas::Transpose::Trans;
        lda = incA;
    }

    if (batch_size == 1 && n == 1) {
        auto layout = transA == gblas::Transpose::NoTrans
            ? gblas::Layout::RowMajor
            : gblas::Layout::ColMajor;
        gblas::gemv(layout, gblas::Transpose::NoTrans,
                    m, k,
                    T{1},
                    A, shapeA.offset(), lda,
                    B, shapeB.offset(), ldb,
                    T{0},
                    C, 0, 1);
        return;
    }

    auto transB = gblas::Transpose::NoTrans;
    if (incB != 1) {
        assert(ldb == 1);
        transB = gblas::Transpose::Trans;
        ldb = incB;
    }

    if (batch_size == 1) {
        gblas::gemm(
            gblas::Layout::RowMajor, transA, transB,
            m, n, k,
            T{1},
            A, shapeA.offset(), lda,
            B, shapeB.offset(), ldb,
            T{0},
            C, 0, ldc);
    } else if (is_contiguous_strides(shapeA) && is_contiguous_strides(shapeB)) {
        gblas::gemmStridedBatched(
            gblas::Layout::RowMajor, transA, transB,
            m, n, k,
            T{1},
            A, shapeA.offset(), lda, shapeA.stride(-3),
            B, shapeB.offset(), ldb, shapeB.stride(-3),
            T{0},
            C, 0, ldc, m*n,
            batch_size);
    } else {
        std::vector<T> alphas(batch_size);
        std::vector<T> betas(batch_size);
        std::vector<size_t> a_offsets(batch_size);
        std::vector<size_t> b_offsets(batch_size);
        std::vector<size_t> c_offsets(batch_size);

        for (int p = 0; p < batch_size; p++) {
            alphas[p]    = T{1};
            betas[p]     = T{0};
            a_offsets[p] = batch_offset(shapeA, p);
            b_offsets[p] = batch_offset(shapeB, p);
            c_offsets[p] = p*m*n;
        }

        gblas::gemmBatched(
            gblas::Layout::RowMajor, transA, transB,
            m, n, k,
            &alphas[0],
            A, &a_offsets[0], lda,
            B, &b_offsets[0], ldb,
            &betas[0],
            C, &c_offsets[0], ldc,
            batch_size);
    }
}
} // namespace detail

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
template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>&>
matmul(const LHS& A, const RHS& B, tensor_type<LHS>& C) {
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
    int ldc = shapeC.stride(-2);
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

    if (m != 1 || n != 1) {
        bool reordered = false;
        if ((incA != 1 && lda != 1) || (incA == 0 || lda == 0)) {
            reorder(A, tmpA);
            shapeA = tmpA.shape();
            dataA = tmpA.data();
            lda = k;
            incA = 1;
            reordered = true;
        }
        if ((incB != 1 && ldb != 1) || (incB == 0 || ldb == 0)) {
            reorder(B, tmpB);
            shapeB = tmpB.shape();
            dataB = tmpB.data();
            ldb = n;
            incB = 1;
            reordered = true;
        }
        if (reordered) {
            batch_size = detail::matmul_broadcast(shapeA, shapeB, shapeC);
        }
    }

    // remove 1 from final shape if one of input tensors is a vector
    if (A.rank() == 1)
        shapeC = shapeC.squeeze(-2);
    if (B.rank() == 1)
        shapeC = shapeC.squeeze(-1);
    if (shapeC.rank() == 0)
        shapeC = Shape({1});
    C.resize(shapeC);

    // do batched matrix multiplication
    detail::matmul(
        m, n, k,
        shapeA, dataA, lda, incA,
        shapeB, dataB, ldb, incB,
        C.data(), ldc,
        batch_size);

    return C;
}

template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
inline matmul(const LHS& A, const RHS& B) {
    tensor_type<LHS> C{};
    matmul(A, B, C);
    return C;
}

//==-------------------------------------------------------------------------
// Matrix exponentiation
//==-------------------------------------------------------------------------

template <typename TensorT>
enable_if_tensor<TensorT> matpow(TensorT&& A, long n) {
    if (!A.is_inner_square())
        throw shape_error("matpow: square matrix required");
    if (n < 0)
        throw std::logic_error("matpow: negative exponentiation is not supported");

    if (n == 0)
        return Tensor<tensor_value_type<TensorT>>::identity(2, A.extent(0)); // FIXME
    if (n == 1)
        return std::forward<TensorT>(A);
    n--;

    tensor_type<TensorT> x = std::forward<TensorT>(A);
    auto y = x, t = x;
    while (n > 0) {
        if (n & 1)
            std::swap(y, matmul(x, y, t));
        std::swap(x, matmul(x, x, t));
        n >>= 1;
    }
    return y;
}

//==-------------------------------------------------------------------------
// Dot product
//==-------------------------------------------------------------------------

template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
dot(const LHS& A, const RHS& B) {
    if (A.rank() > 2 || B.rank() > 2)
        throw shape_error("dot: unsupported tensor shape");
    tensor_type<LHS> C{};
    matmul(A, B, C);
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
// Cross product
//==-------------------------------------------------------------------------

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
inline cross(const LHS& lhs, const RHS& rhs) {
    return cross(lhs, rhs, xfn::multiplies<>());
}

//==-------------------------------------------------------------------------
// Linear algebra operations
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
enable_if_tensor<TensorT> trace(const TensorT& X, int offset = 0, int axis1 = -2, int axis2 = -1) {
    return reduce_sum(X.diagonal(offset, axis1, axis2), {-1}, false);
}

} // namespace dlf
