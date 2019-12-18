#pragma once

namespace dlf { namespace detail {

//==-------------------------------------------------------------------------
// Low level BLAS routines
//==-------------------------------------------------------------------------

template <typename T>
void vscal(const int n, const T& alpha, T* x, const int x_inc) {
    if (alpha == xfn::one<T>())
        return;
    if (alpha == xfn::zero<T>()) {
        if (x_inc == 1)
            std::fill(x, x + n, alpha);
        else {
            for (int i = 0; i < n; ++i, x += x_inc)
                *x = alpha;
        }
    } else {
        if (x_inc == 1)
            std::transform(x, x + n, x, [&alpha](const auto& x){ return alpha * x; });
        else {
            for (int i = 0; i < n; ++i, x += x_inc)
                *x = alpha * *x;
        }
    }
}

template <typename T>
T vdot(const int n, T init, const T* x, const int x_inc, const T* y, const int y_inc) {
    if (x_inc == 1 && y_inc == 1) {
        for (int i = 0; i < n; ++i, ++x, ++y)
            init += *x * *y;
    } else {
        for (int i = 0; i < n; ++i, x += x_inc, y += y_inc)
            init += *x * *y;
    }
    return init;
}

template <typename T>
void vmad(const int n, const T& alpha, const T* x, const int x_inc, T* y, const int y_inc) {
    if (n <= 0 || alpha == xfn::zero<T>())
        return;
    if (x_inc == 1 && y_inc == 1) {
        for (int i = 0; i < n; ++i, ++x, ++y)
            *y += alpha * *x;
    } else {
        for (int i = 0; i < n; ++i, x += x_inc, y += y_inc)
            *y += alpha * *x;
    }
}

template <typename T>
void gemv_serial(cblas::Transpose trans, int m, int n,
                 const T& alpha, const T* A, int lda, const T* x, int incX,
                 const T& beta, T* y, int incY)
{
    if (trans == cblas::Transpose::NoTrans) {
        for (int i = 0; i < m; ++i, A += lda, y += incY) {
            auto acc = vdot(n, xfn::zero<T>(), A, 1, x, incX);
            *y = alpha * acc + beta * *y;
        }
    } else {
        vscal(n, beta, y, incY);
        if (alpha == xfn::zero<T>())
            return;
        for (int i = 0; i < m; ++i, x += incX, A += lda)
            vmad(n, alpha * *x, A, 1, y, incY);
    }
}

template <typename T>
struct gemv_task : tbb::task {
    int m, n;
    const T* A; int lda;
    const T* x; int incX;
    std::vector<T>& v;

    gemv_task(int m, int n, const T* A, int lda, const T* x, int incX, std::vector<T>& v)
        : m(m), n(n), A(A), lda(lda), x(x), incX(incX), v(v)
    {}

    task* execute() override;

    static void run(int m, int n,
                    const T& alpha, const T* A, int lda, const T* x, int incX,
                    const T& beta, T* y, int incY)
    {
        auto v = std::vector<T>(n, xfn::zero<T>());
        auto root = new(allocate_root()) gemv_task(m, n, A, lda, x, incX, v);
        spawn_root_and_wait(*root);
        for (int i = 0; i < n; ++i, y += incY) {
            *y = alpha * v[i] + beta * *y;
        }
    }
};

template <typename T>
struct gemv_merge_task : tbb::task {
    std::vector<T>& y;
    std::vector<T> x;
    gemv_merge_task(int n, std::vector<T>& y)
        : y(y), x(n, xfn::zero<T>()) {}
    task* execute() override {
        std::transform(y.begin(), y.end(), x.begin(), y.begin(), std::plus<>());
        return nullptr;
    }
};

template <typename T>
tbb::task* gemv_task<T>::execute() {
    if (m <= 128) {
        gemv_serial(cblas::Transpose::Trans, m, n,
                    xfn::one<T>(), A, lda, x, incX,
                    xfn::zero<T>(), v.data(), 1);
        return nullptr;
    } else {
        auto mid = m / 2;
        auto c = new(allocate_continuation()) gemv_merge_task<T>(n, v);
        c->set_ref_count(2);
        spawn(*new(c->allocate_child()) gemv_task(
            m - mid, n, A + mid*lda, lda, x + mid*incX, incX, c->x));
        m = mid;
        recycle_as_child_of(*c);
        return this;
    }
}

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

    if (trans == cblas::Transpose::NoTrans) {
        tbb::parallel_for(tbb::blocked_range<int>(0, m), [=](auto r) {
            gemv_serial(trans, r.size(), n, alpha,
                        A + r.begin()*lda, lda, x, incX,
                        beta, y + r.begin()*incY, incY);
        });
    } else {
        gemv_task<T>::run(m, n, alpha, A, lda, x, incX, beta, y, incY);
    }
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
void gemm_ikj(const int m, const int n, const int p,
              const T& alpha,
              const T* A, int lda, int incA,
              const T* B, int ldb, int incB,
              const T& beta, T* C, const int ldc)
{
    for (int i = 0; i < m; ++i, A += lda, C += ldc) {
        if (beta == xfn::zero<T>())
            std::fill(C, C+n, beta);
        else if (beta != xfn::one<T>())
            std::transform(C, C+n, C, [&](const auto& x){ return beta*x; });
        if (alpha == xfn::zero<T>())
            continue;

        auto pa = A, pb = B;
        for (int k = 0; k < p; ++k, pa += incA, pb += ldb) {
            vmad(n, alpha * *pa, pb, incB, C, 1);
        }
    }
}

template <typename T>
void gemm_ijk(const int m, const int n, const int p,
              const T& alpha,
              const T* A, int lda, int incA,
              const T* B, int ldb, int incB,
              const T& beta, T* C, const int ldc)
{
    for (int i = 0; i < m; ++i, A += lda, C += ldc) {
        if (alpha == xfn::zero<T>()) {
            if (beta == xfn::zero<T>())
                std::fill(C, C+n, beta);
            else if (beta != xfn::one<T>())
                std::transform(C, C+n, C, [&](const auto& x){ return beta*x; });
            continue;
        }

        for (int j = 0; j < n; ++j) {
            auto acc = vdot(p, xfn::zero<T>(), A, incA, B + j*incB, ldb);
            C[j] = alpha * acc + beta * C[j];
        }
    }
}

template <typename T>
void gemm_serial(const int m, const int n, const int p,
                 const T& alpha,
                 const T* A, int lda, int incA,
                 const T* B, int ldb, int incB,
                 const T& beta, T* C, const int ldc)
{
    if (incB < ldb) {
        gemm_ikj(m, n, p, alpha, A, lda, incA, B, ldb, incB, beta, C, ldc);
    } else {
        gemm_ijk(m, n, p, alpha, A, lda, incA, B, ldb, incB, beta, C, ldc);
    }
}

template <typename T>
void gemm_direct(const int m, const int n, const int p,
                 const T& alpha,
                 const T* A, int lda, int incA,
                 const T* B, int ldb, int incB,
                 const T& beta,
                 T* C, const int ldc)
{
    if (m == 1 && n == 1) {
        if (alpha == xfn::zero<T>()) {
            *C = beta * *C;
            return;
        }
        auto v = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, p, GRAINSIZE),
            xfn::zero<T>(),
            [=](auto r, T sum) {
                return vdot(r.size(), sum, A + r.begin()*incA, incA, B + r.begin()*ldb, ldb);
            },
            std::plus<T>());
        *C = alpha * v + beta * *C;
        return;
    }

    if (n == 1 && (incA == 1 || lda == 1)) {
        if (incA == 1)
            gemv(cblas::Transpose::NoTrans, m, p, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            gemv(cblas::Transpose::Trans, p, m, alpha, A, incA, B, ldb, beta, C, ldc);
        return;
    }

    if (m * n < GRAINSIZE) {
        gemm_serial(m, n, p, alpha, A, lda, incA, B, ldb, incB, beta, C, ldc);
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0, m), [=, &alpha, &beta](const auto& r) {
            gemm_serial(r.size(), n, p, alpha,
                        A + r.begin()*lda, lda, incA,
                        B, ldb, incB,
                        beta, C + r.begin()*ldc, ldc);
        });
    }
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
    gemm_direct(m, n, p, alpha, A, lda, incA, B, ldb, incB, beta, C, ldc);
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
void symv_up(const int n,
             const T& alpha, const T* A, int lda, const T* x, int incX,
             const T& beta, T* y, int incY)
{
    auto pa = A, px = x;
    auto py = y;
    int i;

    for (i = 0; i < n; ++i, pa += lda, px += incX, py += incY) {
        auto acc = vdot(n - i, xfn::zero<T>(), pa + i, 1, px, incX);
        *py = alpha * acc + beta * *py;
    }
    for (i = 1, y += incY; i < n; ++i, A += lda, x += incX, y += incY) {
        vmad(n - i, alpha * *x, A + i, 1, y, incY);
    }
}

template <typename T>
void symv_lo(const int n,
             const T& alpha, const T* A, int lda, const T* x, int incX,
             const T& beta, T* y, int incY)
{
    auto pa = A;
    auto py = y;
    int i;

    for (i = 0; i < n; ++i, pa += lda, py += incY) {
        auto acc = vdot(i+1, xfn::zero<T>(), pa, 1, x, incX);
        *py = alpha * acc + beta * *py;
    }
    for (i = 1, A += lda, x += incX; i < n; ++i, A += lda, x += incX) {
        vmad(i, alpha * *x, A, 1, y, incY);
    }
}

template <typename T>
std::enable_if_t<!(std::is_same<T, float>::value || std::is_same<T, double>::value)>
symv(cblas::Triangle uplo, const int n,
     const T& alpha, const T* A, int lda, const T* x, int incX,
     const T& beta, T* y, int incY)
{
    if (uplo == cblas::Triangle::Upper) {
        symv_up(n, alpha, A, lda, x, incX, beta, y, incY);
    } else {
        symv_lo(n, alpha, A, lda, x, incX, beta, y, incY);
    }
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
    if (side == cblas::Side::Left) {
        tbb::parallel_for(tbb::blocked_range<int>(0, n), [=](const auto& r) {
            std::vector<T> x(m), y(m);
            for (int i = r.begin(); i < r.end(); ++i) {
                for (int j = 0; j < m; ++j) {
                    x[j] = B[j*ldb + i];
                    y[j] = C[j*ldc + i];
                }
                detail::symv(uplo, m, alpha, A, lda, x.data(), 1, beta, y.data(), 1);
                for (int j = 0; j < m; ++j)
                    C[j*ldc + i] = y[j];
            }
        });
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0, m), [=](const auto& r) {
            for (int i = r.begin(); i < r.end(); ++i) {
                detail::symv(uplo, n, alpha, A, lda, B + i*ldb, 1, beta, C + i*ldc, 1);
            }
        });
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
void trmv_up(const int n, const T* A, int lda, T* x, int incX, bool trans, bool unit) {
    if (!trans) {
        for (int i = 0; i < n; ++i, A += lda, x += incX) {
            if (unit) {
                *x = vdot(n-i-1, *x, A+i+1, 1, x+incX, incX);
            } else {
                *x = vdot(n-i, xfn::zero<T>(), A+i, 1, x, incX);
            }
        }
    } else {
        A += (n - 1) * lda;
        x += (n - 1) * incX;
        for (int i = n; --i >= 0; A -= lda, x -= incX) {
            vmad(n-i-1, *x, A+i+1, 1, x+incX, incX);
            if (!unit) *x *= A[i];
        }
    }
}

template <typename T>
void trmv_lo(const int n, const T* A, int lda, T* x, int incX, bool trans, bool unit) {
    if (!trans) {
        auto y = x + (n - 1) * incX;
        A += (n - 1) * lda;
        for (int i = n; --i >= 0; A -= lda, y -= incX) {
            if (unit) {
                *y = vdot(i, *y, A, 1, x, incX);
            } else {
                *y = vdot(i+1, xfn::zero<T>(), A, 1, x, incX);
            }
        }
    } else {
        auto y = x;
        for (int i = 0; i < n; ++i, A += lda, x += incX) {
            vmad(i, *x, A, 1, y, incX);
            if (!unit) *x *= A[i];
        }
    }
}

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
trmv(cblas::Triangle uplo, cblas::Transpose trans, cblas::Diagonal diag,
     const int n, const T* A, int lda, T* x, int incX)
{
    if (uplo == cblas::Triangle::Upper) {
        trmv_up(n, A, lda, x, incX,
                trans != cblas::Transpose::NoTrans,
                diag  == cblas::Diagonal::Unit);
    } else {
        trmv_lo(n, A, lda, x, incX,
                trans != cblas::Transpose::NoTrans,
                diag  == cblas::Diagonal::Unit);
    }
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
void trsv_up(const int n, const T* A, const int lda, T* x, const int incX, bool trans, bool nounit) {
    if (!trans) {
        A += (n - 1)*lda;
        x += (n - 1)*incX;
        for (int i = n; --i >= 0; A -= lda, x -= incX) {
            *x -= detail::vdot(n-i-1, xfn::zero<T>(), A+i+1, 1, x+incX, incX);
            if (nounit) *x /= A[i];
        }
    } else {
        for (int i = 0; i < n; ++i, A += lda, x += incX) {
            if (nounit) *x /= A[i];
            detail::vmad(n-i-1, -*x, A+i+1, 1, x+incX, incX);
        }
    }
}

template <typename T>
void trsv_lo(const int n, const T* A, const int lda, T* x, const int incX, bool trans, bool nounit) {
    if (!trans) {
        auto y = x;
        for (int i = 0; i < n; ++i, A += lda, y += incX) {
            *y -= detail::vdot(i, xfn::zero<T>(), A, 1, x, incX);
            if (nounit) *y /= A[i];
        }
    } else {
        auto y = x;
        A += (n - 1)*lda;
        y += (n - 1)*incX;
        for (int i = n; --i >= 0; A -= lda, y -= incX) {
            if (nounit) *y /= A[i];
            detail::vmad(i, -*y, A, 1, x, incX);
        }
    }
}

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
trsv(cblas::Triangle uplo, cblas::Transpose trans, cblas::Diagonal diag,
     const int n, const T* A, const int lda, T* x, const int incX)
{
    if (uplo == cblas::Triangle::Upper) {
        trsv_up(n, A, lda, x, incX,
                trans != cblas::Transpose::NoTrans,
                diag  == cblas::Diagonal::NonUnit);
    } else {
        trsv_lo(n, A, lda, x, incX,
                trans != cblas::Transpose::NoTrans,
                diag  == cblas::Diagonal::NonUnit);
    }
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value>
inline trsv(cblas::Triangle uplo, cblas::Transpose trans, cblas::Diagonal diag,
            const int n, const T* A, const int lda, T* x, const int incX)
{
    cblas::trsv(cblas::Layout::RowMajor, uplo, trans, diag, n, A, lda, x, incX);
}

template <typename T>
inline void trsv(cblas::Triangle uplo, cblas::Transpose trans, cblas::Diagonal diag,
                 const int n, const gpgpu::Buffer<T>& A, int lda,
                 gpgpu::Buffer<T>& x, int incX)
{
    gblas::trsv(gblas::Layout::RowMajor,
                static_cast<gblas::Triangle>(uplo),
                static_cast<gblas::Transpose>(trans),
                static_cast<gblas::Diagonal>(diag),
                n, A, 0, lda, x, 0, incX);
}

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
trmm(cblas::Side side, cblas::Triangle uplo, cblas::Transpose transA, cblas::Diagonal diag,
     const int m, const int n, const T& alpha,
     const T* A, const int lda, T* B, const int ldb)
{
    if (alpha == xfn::zero<T>()) {
        for (int i = 0; i < m; ++i, B += ldb)
            std::fill(B, B + n, alpha);
        return;
    }

    if (alpha != xfn::one<T>()) {
        for (int i = 0; i < m; ++i, B += ldb) {
            std::transform(B, B + n, B, [&alpha](const auto& x){ return alpha * x; });
        }
    }

    if (side == cblas::Side::Left) {
        tbb::parallel_for(tbb::blocked_range<int>(0, n), [=](const auto& r) {
            std::vector<T> x(m);
            for (int i = r.begin(); i < r.end(); ++i) {
                for (int j = 0; j < m; ++j)
                    x[j] = B[j*ldb + i];
                detail::trmv(uplo, transA, diag, m, A, lda, x.data(), 1);
                for (int j = 0; j < m; ++j)
                    B[j*ldb + i] = x[j];
            }
        });
    } else {
        transA = transA == cblas::Transpose::NoTrans ? cblas::Transpose::Trans : cblas::Transpose::NoTrans;
        tbb::parallel_for(tbb::blocked_range<int>(0, m), [=](const auto& r) {
            for (int i = r.begin(); i < r.end(); ++i) {
                detail::trmv(uplo, transA, diag, n, A, lda, B + i*ldb, 1);
            }
        });
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

template <typename T>
std::enable_if_t<!cblas::is_blasable<T>::value>
trsm(cblas::Side side, cblas::Triangle uplo, cblas::Transpose transA, cblas::Diagonal diag,
     const int m, const int n, const T& alpha,
     const T* A, int lda, T* B, int ldb)
{
    if (alpha == xfn::zero<T>()) {
        for (int i = 0; i < m; ++i, B += ldb)
            std::fill(B, B + n, alpha);
        return;
    }

    if (alpha != xfn::one<T>()) {
        for (int i = 0; i < m; ++i, B += ldb) {
            std::transform(B, B + n, B, [&alpha](const auto& x){ return alpha * x; });
        }
    }

    if (side == cblas::Side::Left) {
        tbb::parallel_for(tbb::blocked_range<int>(0, n), [=](const auto& r) {
            std::vector<T> x(m);
            for (int i = r.begin(); i < r.end(); ++i) {
                for (int j = 0; j < m; ++j)
                    x[j] = B[j*ldb + i];
                trsv(uplo, transA, diag, m, A, lda, x.data(), 1);
                for (int j = 0; j < m; ++j)
                    B[j*ldb + i] = x[j];
            }
        });
    } else {
        transA = transA == cblas::Transpose::NoTrans ? cblas::Transpose::Trans : cblas::Transpose::NoTrans;
        tbb::parallel_for(tbb::blocked_range<int>(0, m), [=](const auto& r) {
            for (int i = r.begin(); i < r.end(); ++i) {
                trsv(uplo, transA, diag, n, A, lda, B + i*ldb, 1);
            }
        });
    }
}

template <typename T>
std::enable_if_t<cblas::is_blasable<T>::value>
inline trsm(cblas::Side side, cblas::Triangle uplo, cblas::Transpose transA, cblas::Diagonal diag,
            const int m, const int n, const T& alpha,
            const T* A, int lda, T* B, int ldb)
{
    cblas::trsm(cblas::Layout::RowMajor, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <typename T>
inline void trsm(cblas::Side side, cblas::Triangle uplo, cblas::Transpose transA, cblas::Diagonal diag,
                 const int m, const int n, const T& alpha,
                 const gpgpu::Buffer<T>& A, const int lda,
                 gpgpu::Buffer<T>& B, const int ldb)
{
    gblas::trsm(gblas::Layout::RowMajor,
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
    shapeA = shapeA.broadcast_to(Shape(dimsA));

    dimsB = prefixShape.extents();
    dimsB.push_back(k);
    dimsB.push_back(n);
    shapeB = shapeB.broadcast_to(Shape(dimsB));

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

template <typename LHS, typename RHS>
std::enable_if_t<
    (is_gpu_tensor<LHS>::value || cblas::is_blasable<tensor_value_type<LHS>>::value) &&
    (is_gpu_tensor<RHS>::value || cblas::is_blasable<tensor_value_type<RHS>>::value)>
inline is_matmul_reorder_needed(bool* reorderA, bool* reorderB,
                                int m, int n, int k,
                                int lda, int incA,
                                int ldb, int incB)
{
    *reorderA = is_matmul_lhs_need_reorder<LHS>(m, n, k, lda, incA);
    *reorderB = is_matmul_rhs_need_reorder<RHS>(k, n, ldb, incB);
}

template <typename LHS, typename RHS>
std::enable_if_t<
    (is_cpu_tensor<LHS>::value && !cblas::is_blasable<tensor_value_type<LHS>>::value) ||
    (is_cpu_tensor<RHS>::value && !cblas::is_blasable<tensor_value_type<RHS>>::value)>
inline is_matmul_reorder_needed(bool* reorderA, bool* reorderB,
                                int m, int n, int,
                                int lda, int incA,
                                int ldb, int incB)
{
    *reorderA = false;
    *reorderB = (m * n >= GRAINSIZE) && (incA > lda) && (incB > ldb);
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
inline matmul_cpu(int m, int n, int k,
                  const T& alpha,
                  const T* A, int lda, int incA,
                  const T* B, int ldb, int incB,
                  const T& beta, T* C, int ldc)
{
    gemm_direct(m, n, k, alpha, A, lda, incA, B, ldb, incB, beta, C, ldc);
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
