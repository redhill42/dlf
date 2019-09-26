#pragma once

namespace dlf { namespace detail {

//==-------------------------------------------------------------------------
// Low level BLAS routines
//==-------------------------------------------------------------------------

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
gemv(cblas::Transpose trans, int m, int n,
     const T& alpha, const T* A, int lda, const T* x, int incX,
     const T& beta, T* y, int incY)
{
    if (x == y) {
        auto z = std::make_unique<T[]>(n);
        for (int i = 0; i < n; ++i)
            z[i] = x[i*incX];
        gemv(trans, m, n, alpha, A, lda, z.get(), 1, beta, y, incY);
        return;
    }

    int incA = 1;
    if (trans != cblas::Transpose::NoTrans) {
        std::swap(lda, incA);
    }

    tbb::parallel_for(tbb::blocked_range<int>(0, m, 32), [=](auto r) {
        for (int i = r.begin(); i < r.end(); i++) {
            auto acc = y[i*incY] * beta;
            auto pa = A + i*lda;
            auto px = x;
            for (int j = 0; j < n; j++, pa += incA, px += incX)
                acc += alpha * *pa * *px;
            y[i*incY] = acc;
        }
    });
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value>
inline gemv(cblas::Transpose trans, int m, int n,
            const T& alpha, const T* A, int lda, const T* x, int incX,
            const T& beta, T* y, int incY)
{
    cblas::gemv(cblas::Layout::RowMajor, trans, m, n, alpha, A, lda, x, incX, beta, y, incY);
}

template <typename T>
inline void gemv(cblas::Transpose trans, int m, int n,
                 const T& alpha,
                 const gpgpu::Buffer<T>& A, int lda,
                 const gpgpu::Buffer<T>& x, int incX,
                 const T& beta, gpgpu::Buffer<T>& y, int incY)
{
    gblas::gemv(gblas::Layout::RowMajor,
                static_cast<gblas::Transpose>(trans),
                m, n, alpha, A, lda, x, incX, beta, y, incY);
}

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
gemm(cblas::Transpose transA, cblas::Transpose transB,
     const int m, const int n, const int p,
     const T& alpha,
     const T* A, int lda,
     const T* B, int ldb,
     const T& beta,
     T* C, const int ldc)
{
    int incA = 1, incB = 1;
    if (transA != cblas::Transpose::NoTrans)
        std::swap(lda, incA);
    if (transB != cblas::Transpose::NoTrans)
        std::swap(ldb, incB);

    if (beta == xfn::zero<T>()) {
        par::fill(C, C + m*n, beta);
    } else if (beta != xfn::one<T>()) {
        par::transform(C, C + m*n, C, [&beta](const auto& x){ return beta*x; });
    }
    if (alpha == xfn::zero<T>()) {
        return;
    }

    tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 32, 0, n, 32), [=](auto r) {
        const int i0   = r.rows().begin();
        const int j0   = r.cols().begin();
        const int rows = r.rows().size();
        const int cols = r.cols().size();

        auto pa0 = A + i0*lda;
        auto pb0 = B + j0*incB;
        auto pc  = C + i0*ldc + j0;

        for (int i = 0; i < rows; i++, pa0 += lda, pc += ldc) {
            auto pa = pa0, pb = pb0;
            for (int k = 0; k < p; k++, pa += incA, pb += ldb) {
                const auto temp = alpha * *pa;
                for (int j = 0; j < cols; j++, pb += incB)
                    pc[j] += temp * *pb;
                pb -= cols*incB;
            }
        }
    });
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value>
inline gemm(cblas::Transpose transA, cblas::Transpose transB,
            const int m, const int n, const int k,
            const T& alpha,
            const T* A, const int lda,
            const T* B, const int ldb,
            const T& beta,
            T* C, const int ldc)
{
    cblas::gemm(cblas::Layout::RowMajor, transA, transB,
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
inline void gemm(cblas::Transpose transA, cblas::Transpose transB,
                 const int m, const int n, const int k,
                 const T& alpha,
                 const gpgpu::Buffer<T>& A, const int lda,
                 const gpgpu::Buffer<T>& B, const int ldb,
                 const T& beta,
                 gpgpu::Buffer<T>& C, const int ldc)
{
    gblas::gemm(gblas::Layout::RowMajor,
                static_cast<gblas::Transpose>(transA),
                static_cast<gblas::Transpose>(transB),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

//==-------------------------------------------------------------------------
// symmetric matrix multiplication
//==-------------------------------------------------------------------------

template <typename T>
void symmetric_lower_to_squared(const T* A, int lda, T* X, int k) {
    tbb::parallel_for(tbb::blocked_range2d<int>(0, k, 32, 0, k, 32), [=](auto r) {
        for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
            T* px = X + i*k + r.cols().begin();
            for (int j = r.cols().begin(); j < r.cols().end(); j++, px++) {
                *px = (j <= i) ? A[i*lda + j] : A[j*lda + i];
            }
        }
    });
}

template <typename T>
void symmetric_upper_to_squared(const T* A, int lda, T* X, int k) {
    tbb::parallel_for(tbb::blocked_range2d<int>(0, k, 32, 0, k, 32), [=](auto r) {
        for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
            T* px = X + i*k + r.cols().begin();
            for (int j = r.cols().begin(); j < r.cols().end(); j++, px++) {
                *px = (j >= i) ? A[i*lda + j] : A[j*lda + i];
            }
        }
    });
}

template <typename T>
std::enable_if_t<!(std::is_same<T, float>::value || std::is_same<T, double>::value)>
symv(cblas::Triangle uplo, const int n,
     const T& alpha, const T* A, int lda, const T* x, int incX,
     const T& beta, T* y, int incY)
{
    auto t = std::make_unique<T[]>(n * n);
    if (uplo == cblas::Triangle::Lower)
        symmetric_lower_to_squared(A, lda, t.get(), n);
    else
        symmetric_upper_to_squared(A, lda, t.get(), n);
    gemv(cblas::Transpose::NoTrans, n, n, alpha, t.get(), n, x, incX, beta, y, incY);
}

template <typename T>
std::enable_if_t<std::is_same<T, float>::value || std::is_same<T, double>::value>
inline symv(cblas::Triangle uplo, const int n,
            const T& alpha, const T* A, int lda, const T* x, int incX,
            const T& beta, T* y, int incY)
{
    cblas::symv(cblas::Layout::RowMajor, uplo, n, alpha, A, lda, x, incX, beta, y, incY);
}

template <typename T>
inline void symv(cblas::Triangle uplo, const int n,
                 const T& alpha, const gpgpu::Buffer<T>& A, int lda,
                 const gpgpu::Buffer<T>& x, int incX,
                 const T& beta, gpgpu::Buffer<T>& y, int incY)
{
    gblas::symv(gblas::Layout::RowMajor, static_cast<gblas::Triangle>(uplo),
                n, alpha, A, 0, lda, x, 0, incX, beta, y, 0, incY);
}

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
symm(cblas::Side side, cblas::Triangle uplo, const int m, const int n,
     const T& alpha, const T* A, int lda, const T* B, int ldb,
     const T& beta, T* C, const int ldc)
{
    auto k = (side == cblas::Side::Left) ? m : n;
    auto X = std::make_unique<T[]>(k * k);
    if (uplo == cblas::Triangle::Lower)
        symmetric_lower_to_squared(A, lda, X.get(), k);
    else
        symmetric_upper_to_squared(A, lda, X.get(), k);

    if (side == cblas::Side::Left) {
        gemm(cblas::Transpose::NoTrans, cblas::Transpose::NoTrans,
             m, n, k, alpha, X.get(), k, B, ldb, beta, C, ldc);
    } else {
        gemm(cblas::Transpose::NoTrans, cblas::Transpose::NoTrans,
             m, n, k, alpha, B, ldb, X.get(), k, beta, C, ldc);
    }
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value>
inline symm(cblas::Side side, cblas::Triangle uplo, const int m, int n,
            const T& alpha, const T* A, int lda, const T* B, int ldb,
            const T& beta, T* C, const int ldc)
{
    cblas::symm(cblas::Layout::RowMajor, side, uplo, m, n,
                alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
inline void symm(cblas::Side side, cblas::Triangle uplo, const int m, const int n,
                 const T& alpha,
                 const gpgpu::Buffer<T>& A, const int lda,
                 const gpgpu::Buffer<T>& B, const int ldb,
                 const T& beta,
                 gpgpu::Buffer<T>& C, const int ldc)
{
    gblas::symm(gblas::Layout::RowMajor,
                static_cast<gblas::Side>(side),
                static_cast<gblas::Triangle>(uplo),
                m, n, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
}

//==-------------------------------------------------------------------------
// Triangular matrix multiplication
//==-------------------------------------------------------------------------

template <typename T>
void triangular_lower_to_squared(const T* A, int lda, T* X, int k, bool unit_diagonal) {
    tbb::parallel_for(tbb::blocked_range2d<int>(0, k, 32, 0, k, 32), [=](auto r) {
        for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
            T* px = X + i*k + r.cols().begin();
            for (int j = r.cols().begin(); j < r.cols().end(); j++, px++) {
                if (unit_diagonal && i == j)
                    *px = xfn::one<T>();
                else if (j <= i)
                    *px = A[i*lda + j];
                else
                    *px = xfn::zero<T>();
            }
        }
    });
}

template <typename T>
void triangular_upper_to_squared(const T* A, int lda, T* X, int k, bool unit_diagonal) {
    tbb::parallel_for(tbb::blocked_range2d<int>(0, k, 32, 0, k, 32), [=](auto r) {
        for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
            T* px = X + i*k + r.cols().begin();
            for (int j = r.cols().begin(); j < r.cols().end(); j++, px++) {
                if (unit_diagonal && i == j)
                    *px = xfn::one<T>();
                else if (j >= i)
                    *px = A[i*lda + j];
                else
                    *px = xfn::zero<T>();
            }
        }
    });
}

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
trmv(cblas::Triangle uplo, cblas::Transpose trans, cblas::Diagonal diag,
     const int n, const T* A, int lda, T* x, int incX)
{
    auto t = std::make_unique<T[]>(n * n);
    if (uplo == cblas::Triangle::Lower)
        triangular_lower_to_squared(A, lda, t.get(), n, diag == cblas::Diagonal::Unit);
    else
        triangular_upper_to_squared(A, lda, t.get(), n, diag == cblas::Diagonal::Unit);
    gemv(trans, n, n, xfn::one<T>(), t.get(), n, x, incX, xfn::zero<T>(), x, incX);
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value>
inline trmv(cblas::Triangle uplo, cblas::Transpose trans, cblas::Diagonal diag,
            const int n, const T* A, int lda, T* x, int incX)
{
    cblas::trmv(cblas::Layout::RowMajor, uplo, trans, diag, n, A, lda, x, incX);
}

template <typename T>
inline void trmv(cblas::Triangle uplo, cblas::Transpose trans, cblas::Diagonal diag,
                 const int n, const gpgpu::Buffer<T>& A, int lda,
                 gpgpu::Buffer<T>& x, int incX)
{
    gblas::trmv(gblas::Layout::RowMajor,
                static_cast<gblas::Triangle>(uplo),
                static_cast<gblas::Transpose>(trans),
                static_cast<gblas::Diagonal>(diag),
                n, A, 0, lda, x, 0, incX);
}

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
trmm(cblas::Side side, cblas::Triangle uplo, cblas::Transpose transA, cblas::Diagonal diag,
     const int m, const int n, const T& alpha,
     const T* A, int lda, T* B, int ldb)
{
    auto k = (side == cblas::Side::Left) ? m : n;
    auto X = std::make_unique<T[]>(k * k);
    if (uplo == cblas::Triangle::Lower)
        triangular_lower_to_squared(A, lda, X.get(), k, diag == cblas::Diagonal::Unit);
    else
        triangular_upper_to_squared(A, lda, X.get(), k, diag == cblas::Diagonal::Unit);

    auto Y = std::make_unique<T[]>(m * n);
    std::copy(B, B+m*n, Y.get());

    if (side == cblas::Side::Left) {
        gemm(transA, cblas::Transpose::NoTrans,
             m, n, k,
             alpha, X.get(), k, Y.get(), n,
             xfn::zero<T>(), B, ldb);
    } else {
        gemm(transA, cblas::Transpose::NoTrans,
             m, n, k,
             alpha, X.get(), k, Y.get(), n,
             xfn::zero<T>(), B, ldb);
    }
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value>
inline trmm(cblas::Side side, cblas::Triangle uplo, cblas::Transpose transA, cblas::Diagonal diag,
            const int m, const int n, const T& alpha,
            const T* A, int lda, T* B, int ldb)
{
    cblas::trmm(cblas::Layout::RowMajor, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <typename T>
inline void trmm(cblas::Side side, cblas::Triangle uplo, cblas::Transpose transA, cblas::Diagonal diag,
                 const int m, const int n, const T& alpha,
                 const gpgpu::Buffer<T>& A, const int lda,
                 gpgpu::Buffer<T>& B, const int ldb)
{
    gblas::trmm(gblas::Layout::RowMajor,
                static_cast<gblas::Side>(side),
                static_cast<gblas::Triangle>(uplo),
                static_cast<gblas::Transpose>(transA),
                static_cast<gblas::Diagonal>(diag),
                m, n, alpha, A, 0, lda, B, 0, ldb);
}

//==-------------------------------------------------------------------------
// Extended matrix multiplication
//==-------------------------------------------------------------------------

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

template <typename LHS>
std::enable_if_t<
    is_gpu_tensor<LHS>::value ||
    cblas::is_blasable<tensor_value_type<LHS>>::value,
    bool>
is_matmul_lhs_need_reorder(int m, int n, int k, int lda, int incA) {
    if (lda == 0 || incA == 0)
        return true;
    if (is_gpu_tensor<LHS>::value && (lda < 0 || incA < 0))
        return true;
    if (m == 1 && n == 1)
        return false;
    if (lda != 1 && incA != 1)
        return true;
    if (incA == 1 && lda < k)
        return true;
    if (lda == 1 && incA < m)
        return true;
    return false;
}

template <typename RHS>
std::enable_if_t<
    is_gpu_tensor<RHS>::value ||
    cblas::is_blasable<tensor_value_type<RHS>>::value,
    bool>
is_matmul_rhs_need_reorder(int k, int n, int ldb, int incB) {
    if (ldb == 0 || incB == 0)
        return true;
    if (is_gpu_tensor<RHS>::value && (ldb < 0 || incB < 0))
        return true;
    if (n == 1)
        return false;
    if (ldb != 1 && incB != 1)
        return true;
    if (incB == 1 && ldb < n)
        return true;
    if (ldb == 1 && incB < k)
        return true;
    return false;
}

template <typename LHS>
std::enable_if_t<
    is_cpu_tensor<LHS>::value &&
    !cblas::is_blasable<tensor_value_type<LHS>>::value,
    bool>
inline constexpr is_matmul_lhs_need_reorder(int, int, int, int, int) {
    return false;
}

template <typename RHS>
std::enable_if_t<
    is_cpu_tensor<RHS>::value &&
    !cblas::is_blasable<tensor_value_type<RHS>>::value,
    bool>
inline constexpr is_matmul_rhs_need_reorder(int, int, int, int) {
    return false;
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value>
matmul_cpu(int m, int n, int k,
           const T& alpha,
           const T* A, int lda, int incA,
           const T* B, int ldb, int incB,
           const T& beta, T* C, int ldc)
{
    if (m == 1 && n == 1) {
        if (incA < 0)
            A += (k-1)*incA;
        if (ldb < 0)
            B += (k-1)*ldb;
        auto v = cblas::dot(k, A, incA, B, ldb);
        *C = alpha * v + beta * *C;
        return;
    }

    auto transA = cblas::Transpose::NoTrans;
    if (incA != 1) {
        assert(lda == 1);
        transA = cblas::Transpose::Trans;
        lda = incA;
    }

    auto transB = cblas::Transpose::NoTrans;
    if (incB != 1) {
        assert(ldb == 1);
        transB = cblas::Transpose::Trans;
        ldb = incB;
    }

    if (n == 1) {
        auto layout = transA == cblas::Transpose::NoTrans
            ? cblas::Layout::RowMajor
            : cblas::Layout::ColMajor;
        if (ldb < 0)
            B += (k-1)*ldb;
        cblas::gemv(layout, cblas::Transpose::NoTrans,
                    m, k,
                    alpha, A, lda, B, ldb,
                    beta, C, ldc);
    } else {
        cblas::gemm(cblas::Layout::RowMajor, transA, transB,
                    m, n, k,
                    alpha, A, lda, B, ldb,
                    beta, C, ldc);
    }
}

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
matmul_cpu(int m, int n, int p,
           const T& alpha,
           const T* A, int lda, int incA,
           const T* B, int ldb, int incB,
           const T& beta, T* C, int ldc)
{
    if (m == 1 && n == 1) {
        auto v = (alpha == xfn::zero<T>()) ? alpha : tbb::parallel_reduce(
            tbb::blocked_range<int>(0, p, GRAINSIZE),
            xfn::zero<T>(),
            [=](auto r, T sum) {
                auto px = A + r.begin()*incA;
                auto py = B + r.begin()*ldb;
                for (int k = r.size(); --k >= 0; px += incA, py += ldb)
                    sum += *px * *py;
                return sum;
            },
            std::plus<T>());
        *C = alpha * v + beta * *C;
    } else {
        if (beta == xfn::zero<T>()) {
            tbb::parallel_for(tbb::blocked_range<int>(0, m, 32), [=](auto r) {
                for (int i = r.begin(); i < r.end(); i++) {
                    auto pc = C + i*ldc;
                    std::fill(pc, pc + n, xfn::zero<T>());
                }
            });
        } else if (beta != xfn::one<T>()) {
            tbb::parallel_for(tbb::blocked_range<int>(0, m, 32), [=,&beta](auto r) {
                for (int i = r.begin(); i < r.end(); i++) {
                    auto pc = C + i*ldc;
                    std::transform(pc, pc+n, pc, [&](const auto& x){ return beta*x; });
                }
            });
        }
        if (alpha == xfn::zero<T>()) {
            return;
        }

        tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 32, 0, n, 32), [=](auto r) {
            const int i0   = r.rows().begin();
            const int j0   = r.cols().begin();
            const int rows = r.rows().size();
            const int cols = r.cols().size();

            auto pa0 = A + i0*lda;
            auto pb0 = B + j0*incB;
            auto pc  = C + i0*ldc + j0;

            for (int i = 0; i < rows; i++, pa0 += lda, pc += ldc) {
                auto pa = pa0, pb = pb0;
                for (int k = 0; k < p; k++, pa += incA, pb += ldb) {
                    const auto temp = alpha * *pa;
                    for (int j = 0; j < cols; j++, pb += incB)
                        pc[j] += temp * *pb;
                    pb -= cols*incB;
                }
            }
        });
    }
}

template <typename T>
void matmul(int m, int n, int k,
            const T& alpha,
            const Shape& shapeA, const T* A, int lda, int incA,
            const Shape& shapeB, const T* B, int ldb, int incB,
            const T& beta,
            const Shape& shapeC, T* C, int ldc,
            int batch_size)
{
    if (batch_size == 1) {
        matmul_cpu(
            m, n, k,
            alpha,
            A + shapeA.offset(), lda, incA,
            B + shapeB.offset(), ldb, incB,
            beta,
            C + shapeC.offset(), ldc);
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0, batch_size, 16), [=](auto r) {
            for (int p = r.begin(); p < r.end(); p++) {
                matmul_cpu(
                    m, n, k,
                    alpha,
                    A + shapeA.linear_offset(p*m*k), lda, incA,
                    B + shapeB.linear_offset(p*k*n), ldb, incB,
                    beta,
                    C + shapeC.linear_offset(p*m*n), ldc);
            }
        });
    }
}

template <typename T>
void matmul(int m, int n, int k,
            const T& alpha,
            const Shape& shapeA, const gpgpu::Buffer<T>& A, int lda, int incA,
            const Shape& shapeB, const gpgpu::Buffer<T>& B, int ldb, int incB,
            const T& beta,
            const Shape& shapeC, gpgpu::Buffer<T>& C, int ldc,
            int batch_size)
{
    if (batch_size == 1 && m == 1 && n == 1 && alpha == xfn::one<T>() && beta == xfn::zero<T>()) {
        gblas::dot(
            k,
            A, shapeA.offset(), incA,
            B, shapeB.offset(), ldb,
            C, shapeC.offset());
        return;
    }

    auto transA = gblas::Transpose::NoTrans;
    if (incA != 1) {
        assert(lda == 1);
        transA = gblas::Transpose::Trans;
        lda = incA;
    }

    auto transB = gblas::Transpose::NoTrans;
    if (incB != 1) {
        assert(ldb == 1);
        transB = gblas::Transpose::Trans;
        ldb = incB;
    }

    if (batch_size == 1 && n == 1) {
        auto layout = transA == gblas::Transpose::NoTrans
            ? gblas::Layout::RowMajor
            : gblas::Layout::ColMajor;
        gblas::gemv(layout, gblas::Transpose::NoTrans,
                    m, k,
                    alpha,
                    A, shapeA.offset(), lda,
                    B, shapeB.offset(), ldb,
                    beta,
                    C, shapeC.offset(), ldc);
    } else if (batch_size == 1) {
        gblas::gemm(
            gblas::Layout::RowMajor, transA, transB,
            m, n, k,
            alpha,
            A, shapeA.offset(), lda,
            B, shapeB.offset(), ldb,
            beta,
            C, shapeC.offset(), ldc);
    } else if (is_contiguous_strides(shapeA) &&
               is_contiguous_strides(shapeB) &&
               is_contiguous_strides(shapeC)) {
        gblas::gemmStridedBatched(
            gblas::Layout::RowMajor, transA, transB,
            m, n, k,
            alpha,
            A, shapeA.offset(), lda, shapeA.stride(-3),
            B, shapeB.offset(), ldb, shapeB.stride(-3),
            beta,
            C, shapeC.offset(), ldc, shapeC.stride(-3),
            batch_size);
    } else {
        std::vector<T> alphas(batch_size);
        std::vector<T> betas(batch_size);
        std::vector<size_t> a_offsets(batch_size);
        std::vector<size_t> b_offsets(batch_size);
        std::vector<size_t> c_offsets(batch_size);

        for (int p = 0; p < batch_size; p++) {
            alphas[p]    = alpha;
            betas[p]     = beta;
            a_offsets[p] = shapeA.linear_offset(p*m*k);
            b_offsets[p] = shapeB.linear_offset(p*k*n);
            c_offsets[p] = shapeC.linear_offset(p*m*n);
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

//==-------------------------------------------------------------------------

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

//==-------------------------------------------------------------------------

template <typename TensorT>
std::enable_if_t<is_cpu_tensor<TensorT>::value, tensor_type<TensorT>>
norm_p(const TensorT& X, float ord, int axis, bool keepdims) {
    using T = tensor_value_type<TensorT>;
    const auto p = static_cast<T>(ord);
    return reduce(X, {axis}, keepdims,
        xfn::zero<T>(),
        [p](const auto& x) {
            return std::pow(std::abs(x), p);
        },
        xfn::plus<T>(),
        [p](auto acc, auto) -> T {
            return std::pow(acc, xfn::one<T>()/p);
        });
}

template <typename TensorT>
std::enable_if_t<!is_cpu_tensor<TensorT>::value, tensor_type<TensorT>>
norm_p(const TensorT& X, float ord, int axis, bool keepdims) {
    using T = tensor_value_type<TensorT>;
    const auto p = static_cast<T>(ord);
    return pow(reduce_sum(pow(abs(X), p), axis, keepdims), xfn::one<T>()/p);
}

}} // namespace dlf::detail
