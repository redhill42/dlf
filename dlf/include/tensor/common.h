#pragma once

#include <unordered_set>

namespace dlf {

//==-------------------------------------------------------------------------
// General matrix multiplication
//==-------------------------------------------------------------------------

namespace detail {
template <typename T>
std::enable_if_t<!cblas::RequireBlasType<T>>
gemm(const int m, const int n, const int k,
     const T& alpha,
     const T* A, const int lda,
     const T* B, const int ldb,
     const T& beta,
     T* C, const int ldc,
     const bool transA = false,
     const bool transB = false,
     Tensor<T>* = nullptr)
{
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, m, 32, 0, n, 32), [&](auto r) {
        size_t incX = transA ? lda : 1;
        size_t incY = transB ? 1 : ldb;
        for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
            T* pz = C + (i*ldc + r.cols().begin());
            for (size_t j = r.cols().begin(); j != r.cols().end(); j++) {
                const T* px = A + (transA ? i : i*lda);
                const T* py = B + (transB ? j*ldb : j);
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
gemm(const int m, const int n, const int k,
     const T& alpha,
     const T* A, const int lda,
     const T* B, const int ldb,
     const T& beta,
     T* C, const int ldc,
     const bool transA = false,
     const bool transB = false,
     Tensor<T>* = nullptr)
{
    cblas::gemm(cblas::Layout::RowMajor,
                transA ? cblas::Transpose::Trans : cblas::Transpose::NoTrans,
                transB ? cblas::Transpose::Trans : cblas::Transpose::NoTrans,
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
void gemm(const int m, const int n, const int k,
          const T& alpha,
          const gpgpu::Buffer<T>& A, const int lda,
          const gpgpu::Buffer<T>& B, const int ldb,
          const T& beta,
          gpgpu::Buffer<T>& C, const int ldc,
          const bool transA = false,
          const bool transB = false,
          DevTensor<T>* work = nullptr)
{
    gblas::gemm(gblas::Layout::RowMajor,
                transA ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
                transB ? gblas::Transpose::Trans : gblas::Transpose::NoTrans,
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                work == nullptr ? nullptr : &work->data());
}
} // namespace detail

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
gemm(const tensor_value_type<TensorT>& alpha,
     const TensorT& A, const TensorT& B,
     const tensor_value_type<TensorT>& beta,
     TensorT* C,
     bool transA = false, bool transB = false,
     TensorT* work = nullptr)
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

    detail::gemm(
        m, n, k,
        alpha,
        A.data(), A.stride(0),
        B.data(), B.stride(0),
        beta,
        C->data(), C->stride(0),
        transA, transB, work);
}

template <typename T>
inline size_t gemmWorkspaceSize(
    const Tensor<T>&, const Tensor<T>&, const Tensor<T>&,
    bool = false, bool = false)
{
    return 0; // API compatible to DevTensor
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

    auto Y = C.broadcast({m, n}).copy();
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
                    A + shapeA.linear_offset(p*m*k), lda, incA,
                    B + shapeB.linear_offset(p*k*n), ldb, incB,
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
            a_offsets[p] = shapeA.linear_offset(p*m*k);
            b_offsets[p] = shapeB.linear_offset(p*k*n);
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
        if (n & 1)
            std::swap(y, matmul(x, y, t));
        std::swap(x, matmul(x, x, t));
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
 *
 * @tparam LHS
 * @tparam RHS
 */
template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
dot(const LHS& A, const RHS& B) {
    if (A.is_scalar() || B.is_scalar())
        return A * B;
    if ((A.rank() <= 2 && B.rank() <= 2) || B.rank() == 1)
        return matmul(A, B);

    std::vector<int> unsq;
    for (int i = 0; i < B.rank()-1; i++) {
        unsq.push_back(i + A.rank() - 1);
    }

    auto C = matmul(unsqueeze(A, unsq), B);
    C.squeeze(-2);
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

namespace detail {
template <typename TensorT>
std::vector<int> get_matrix_chain_dimensions(const std::vector<TensorT>& args) {
    std::vector<int> dims;

    int m, n;
    if (args[0].is_vector()) {
        m = 1;
        n = args[0].extent(0);
    } else if (args[0].is_matrix()) {
        m = args[0].extent(0);
        n = args[0].extent(1);
    } else {
        throw shape_error("multi_dot: the first argument must be a vector or a matrix");
    }
    dims.push_back(m);
    dims.push_back(n);

    for (int i = 1; i < args.size(); ++i) {
        if (args[i].is_matrix()) {
            if (args[i].extent(0) != n)
                throw shape_error("multi_dot: incompatible shape");
            n = args[i].extent(1);
        } else if (i == args.size()-1 && args[i].is_vector()) {
            if (args[i].extent(0) != n)
                throw shape_error("multi_dot: incompatible shape");
            n = 1;
        } else {
            throw shape_error("multi_dot: the arguments in the middle must be matrices");
        }
        dims.push_back(n);
    }

    return dims;
}

template <int = 0>
Tensor<int> matrix_chain_order(const std::vector<int>& dims) {
    auto n = dims.size() - 1;
    Tensor<int> m({n, n}, std::numeric_limits<int>::max());
    Tensor<int> s({n, n}, 0);

    for (int len = 1; len < n; len++) {
        for (int i = 0; i < n - len; i++) {
            int j = i + len;
            for (int k = i; k < j; k++) {
                int cost = m(i, k) + m(k+1, j) + dims[i]*dims[k+1]*dims[j+1];
                if (cost < m(i, j)) {
                    m(i, j) = cost;
                    s(i, j) = k;
                }
            }
        }
    }
    return s;
}

template <typename TensorT>
tensor_type<TensorT> optimal_parenthesizations(
    int n, const Tensor<int>& s, const std::vector<TensorT>& args,
    int i, int j)
{
    if (i == j) {
        return args[i];
    } else {
        return matmul(optimal_parenthesizations(n, s, args, i, s(i, j)),
                      optimal_parenthesizations(n, s, args, s(i, j)+1, j));
    }
}

template <typename TensorT>
inline tensor_type<TensorT> optimal_parenthesizations(
    const Tensor<int>& s, const std::vector<TensorT>& args)
{
    int n = args.size();
    return optimal_parenthesizations(n, s, args, 0, n-1);
}
} // namespace detail

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
inline multi_dot(const std::vector<TensorT>& args) {
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
multi_dot(const First& first, const Rest&... rest) {
    static_assert(sizeof...(rest) > 1, "multi_dot: two few arguments");
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
            if (axes_a[k] < 0)
                axes_a[k] += A.rank();
            if (axes_a[k] < 0 || axes_a[k] >= A.rank())
                throw shape_error("tensordot: axes_a has incorrect value");

            if (axes_b[k] < 0)
                axes_b[k] += B.rank();
            if (axes_b[k] < 0 || axes_b[k] >= B.rank())
                throw shape_error("tensordot: axes_b has incorrect value");

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

    if (out_dims.empty()) {
        out_dims.push_back(1);
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
 */
template <typename LHS, typename RHS>
std::enable_if_t<is_exactly_same_tensor<LHS, RHS>::value, tensor_type<LHS>>
inner(LHS&& A, RHS&& B) {
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
inline outer(LHS&& A, RHS&& B, Fn f) {
    auto rank = A.rank() + B.rank();
    return transform(unsqueeze_right(std::forward<LHS>(A), rank),
                     unsqueeze_left(std::forward<RHS>(B), rank),
                     f);
}

template <typename LHS, typename RHS>
enable_if_tensors<LHS, RHS, xfn::multiplies<>>
inline outer(LHS&& A, RHS&& B) {
    return outer(std::forward<LHS>(A), std::forward<RHS>(B), xfn::multiplies<>());
}

//==-------------------------------------------------------------------------

/**
 * Computes the Kronecker product, a composite tensor made of blocks of the second
 * tensor scaled by the first.
 *
 * The function assumes that the number of dimensions of a and b are the same,
 * if necessary prepending the smallest with ones. If A.shape = (r0,r1,rN) and
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
kronecker(LHS&& A, RHS&& B) {
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

    std::vector<int> axes(rank);
    std::iota(axes.begin(), axes.end(), 0);

    tensor_invoke_result<xfn::multiplies<>, LHS, RHS> C(Shape{c_dims});
    transformTo(unsqueeze_right(unsqueeze_left(std::forward<LHS>(A), rank), rank*2),
                unsqueeze_left(std::forward<RHS>(B), rank*2),
                partition(C, axes, b_dims), xfn::multiplies<>());
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
