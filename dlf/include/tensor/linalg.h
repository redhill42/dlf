#pragma once

namespace dlf {

//==-------------------------------------------------------------------------
// Low level BLAS routines
//==-------------------------------------------------------------------------

namespace detail {
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
} // namespace detail

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
    if (trans != cblas::Transpose::NoTrans)
        std::swap(m, n);
    if (n != x.extent(0))
        throw shape_error("gemv: incompatible shape");
    y.resize(m);
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

namespace detail {
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
} // namespace detail

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

    auto Y = C.broadcast({m, n}).copy();
    gemm(transA, transB, alpha, A, B, beta, &Y);
    return Y;
}

//==-------------------------------------------------------------------------
// symmetric matrix multiplication
//==-------------------------------------------------------------------------

namespace detail {
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
} // namespace detail

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

namespace detail {
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
} // namespace detail

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

    bool reordered = false;
    if (detail::is_matmul_lhs_need_reorder<LHS>(m, n, k, lda, incA)) {
        reorder(A, tmpA);
        shapeA = tmpA.shape();
        dataA = tmpA.data();
        lda = k;
        incA = 1;
        reordered = true;
    }
    if (detail::is_matmul_rhs_need_reorder<RHS>(k, n, ldb, incB)) {
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
enable_if_tensor<TensorT> trace(const TensorT& X, int offset = 0, int axis1 = -2, int axis2 = -1) {
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
namespace detail {
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
} // namespace detail

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

} // namespace dlf
