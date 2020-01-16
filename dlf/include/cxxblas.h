#pragma once

#include <complex>
#include <tbb/tbb.h>

#if HAS_MKL
  #define HAS_LAPACKE 1
  #define lapack_complex_float std::complex<float>
  #define lapack_complex_double std::complex<double>
  #define MKL_Complex8 lapack_complex_float
  #define MKL_Complex16 lapack_complex_double
  #include <mkl.h>
#elif defined(__APPLE__)
  #define HAS_LAPACKE 0
  #include <Accelerate/Accelerate.h>
  #define lapack_int __CLPK_integer
  #define lapack_logical __CLPK_logical
  #define lapack_complex_float __CLPK_complex
  #define lapack_complex_double __CLPK_doublecomplex
#else
  #include <cblas.h>
#endif

namespace cblas {

template <typename T>
using is_blasable = cxx::disjunction<
    std::is_same<std::remove_cv_t<T>, float>,
    std::is_same<std::remove_cv_t<T>, double>,
    std::is_same<std::remove_cv_t<T>, std::complex<float>>,
    std::is_same<std::remove_cv_t<T>, std::complex<double>>>;

enum class Layout {
    RowMajor = ::CblasRowMajor,
    ColMajor = ::CblasColMajor
};

enum class Transpose {
    NoTrans = ::CblasNoTrans,
    Trans = ::CblasTrans,
    ConjTrans = ::CblasConjTrans
};

enum class Triangle {
    Upper = ::CblasUpper,
    Lower = ::CblasLower,
};

enum class Diagonal {
    NonUnit = ::CblasNonUnit,
    Unit = ::CblasUnit
};

enum class Side {
    Left = CblasLeft,
    Right = CblasRight
};

//==-------------------------------------------------------------------------
// Matrix transposition
//==-------------------------------------------------------------------------

template <typename T>
void omatcopy(Transpose trans, size_t m, size_t n, const T* A, size_t lda, T* B, size_t ldb) {
    constexpr size_t NB = 32;
    if (trans == Transpose::NoTrans) {
        tbb::parallel_for<size_t>(0, m, [=](size_t i) {
            std::copy(A + i*lda, A + i*lda + n, B + i*ldb);
        });
    } else {
        tbb::parallel_for<size_t>(0, m, NB, [=](size_t i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t k = 0; k < NB && i + k < m; ++k) {
                    B[j*ldb + (i + k)] = A[(i + k)*lda + j];
                }
            }
        });
    }
}

/**
 * TRIP: Transposing Rectangular matrices In-place and in Parallel
 * http://www3.risc.jku.at/publications/download/risc_5916/main%20v1.0.2.pdf
 */
template <typename T>
class TRIP {
    static void trip(size_t rs, size_t re, size_t cs, size_t ce, T* A, size_t lda);
    static void sq_trans(size_t, size_t, T*, size_t);
    static void sq_swap(size_t, size_t, size_t, size_t, T*, size_t);
    static void merge(size_t, size_t, size_t, size_t, size_t, size_t, T*, size_t);
    static void merger(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, T*, size_t);
    static void split(size_t, size_t, size_t, size_t, size_t, size_t, T*, size_t);
    static void splitr(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, T*, size_t);
    static void reverse(size_t, size_t, size_t, size_t, size_t, T*, size_t);
    static void reverse_ser(size_t, size_t, size_t, size_t, size_t, size_t, T*, size_t);
    static void reverse_par(size_t, size_t, size_t, size_t, size_t, size_t, T*, size_t);
    static void next(size_t*, size_t*, size_t, size_t);
    static void prev(size_t*, size_t*, size_t, size_t);

public:
    static void trip(size_t m, size_t n, T* A) {
        trip(0, m, 0, n, A, n);
    }

    static void trip(size_t n, T* A, size_t lda) {
        sq_trans(0, n, A, lda);
    }
};

template <typename T>
void TRIP<T>::trip(size_t rs, size_t re, size_t cs, size_t ce, T* A, size_t lda) {
    auto m = re - rs;
    auto n = ce - cs;

    if (m == 1 || n == 1)
        return;
    if (m > n) {
        auto rm = (m >= 2*n) ? (rs + re)/2 : rs + n;
        tbb::parallel_invoke(
            [=]{ trip(rs, rm, cs, ce, A, lda); },
            [=]{ trip(rm, re, cs, ce, A, lda); });
        merge(rm - rs, re - rm, rs, re, cs, ce, A, lda);
    } else if (m < n) {
        auto cm = (n >= 2*m) ? (cs + ce)/2 : cs + m;
        tbb::parallel_invoke(
            [=]{ trip(rs, re, cs, cm, A, lda); },
            [=]{ trip(rs, re, cm, ce, A, lda); });
        split(cm - cs, ce - cm, rs, re, cs, ce, A, lda);
    } else {
        sq_trans(0, n, A + rs*lda + cs, lda);
    }
}

template <typename T>
void TRIP<T>::sq_trans(size_t s, size_t e, T* A, size_t lda) {
    if (e - s <= 32) {
        for (size_t r = s; r < e - 1; ++r)
            for (size_t c = r + 1; c < e; ++c) {
                using std::swap;
                swap(A[r*lda + c], A[c*lda + r]);
            }
    } else {
        size_t m = (s + e) / 2;
        tbb::parallel_invoke(
            [=]{ sq_trans(s, m, A, lda); },
            [=]{ sq_trans(m, e, A, lda); },
            [=]{ sq_swap(m, s, e, m, A, lda); });
    }
}

template <typename T>
void TRIP<T>::sq_swap(size_t rs, size_t cs, size_t re, size_t ce, T* A, size_t lda) {
    if (re - rs <= 32 && ce - cs <= 32) {
        for (size_t r = rs; r < re; ++r)
            for (size_t c = cs; c < ce; ++c) {
                using std::swap;
                swap(A[r*lda + c], A[c*lda + r]);
            }
    } else {
        size_t rm = (rs + re) / 2;
        size_t cm = (cs + ce) / 2;
        tbb::parallel_invoke(
            [=]{ sq_swap(rs, cs, rm, cm, A, lda); },
            [=]{ sq_swap(rm, cs, re, cm, A, lda); },
            [=]{ sq_swap(rs, cm, rm, ce, A, lda); },
            [=]{ sq_swap(rm, cm, re, ce, A, lda); });
    }
}

template <typename T>
void TRIP<T>::merge(size_t p, size_t q, size_t rs, size_t re, size_t cs, size_t ce, T* A, size_t lda) {
    merger(p, q, rs, re, cs, ce, 0, (ce - cs)*(re - rs), ce - cs, A, lda);
}

template <typename T>
void TRIP<T>::merger(size_t p, size_t q, size_t rs, size_t re, size_t cs, size_t ce,
                     size_t m0, size_t m1, size_t k, T* A, size_t lda)
{
    if (k == 1) return;

    auto k2 = k / 2;
    auto r0 = m0 + k2*p;
    auto r1 = m0 + k*p + k2*q;
    auto rm = r0 + k2*q;
    auto mm = m0 + k2*(p + q);

    // reverse whole middle part
    reverse(r0, r1, rs, cs, ce, A, lda);

    // reverse left and right of the middle part
    tbb::parallel_invoke(
        [=]{ reverse(r0, rm, rs, cs, ce, A, lda); },
        [=]{ reverse(rm, r1, rs, cs, ce, A, lda); });

    // merge the resulting sub-arrays
    tbb::parallel_invoke(
        [=]{ merger(p, q, rs, re, cs, ce, m0, mm, k2, A, lda); },
        [=]{ merger(p, q, rs, re, cs, ce, mm, m1, k - k2, A, lda); });
}

template <typename T>
void TRIP<T>::split(size_t p, size_t q, size_t rs, size_t re, size_t cs, size_t ce, T* A, size_t lda) {
    return splitr(p, q, rs, re, cs, ce, 0, (ce - cs)*(re - rs), re - rs, A, lda);
}

template <typename T>
void TRIP<T>::splitr(size_t p, size_t q, size_t rs, size_t re, size_t cs, size_t ce,
                     size_t s0, size_t s1, size_t k, T* A, size_t lda)
{
    if (k == 1) return;

    auto k2 = k / 2;
    auto r0 = s0 + k2*p;
    auto r1 = s0 + k*p + k2*q;
    auto rm = s0 + k*p;
    auto sm = s0 + k2*(p + q);

    // split left and right part
    tbb::parallel_invoke(
        [=]{ splitr(p, q, rs, re, cs, ce, s0, sm, k2, A, lda); },
        [=]{ splitr(p, q, rs, re, cs, ce, sm, s1, k - k2, A, lda); });

    // rotate middle part
    reverse(r0, r1, rs, cs, ce, A, lda);

    // rotate left and right part
    tbb::parallel_invoke(
        [=]{ reverse(r0, rm, rs, cs, ce, A, lda); },
        [=]{ reverse(rm, r1, rs, cs, ce, A, lda); });
}

template <typename T>
inline void TRIP<T>::next(size_t* i, size_t* count, size_t p, size_t stride) {
    if (*count == p - 1) {
        *count = 0;
        *i += stride;
    } else {
        ++*count;
        ++*i;
    }
}

template <typename T>
inline void TRIP<T>::prev(size_t* i, size_t* count, size_t p, size_t stride) {
    if (*count == 0) {
        *count = p - 1;
        *i -= stride;
    } else {
        --*count;
        --*i;
    }
}

template <typename T>
void TRIP<T>::reverse_ser(size_t m0, size_t m1, size_t l,
                          size_t rs, size_t cs, size_t ce,
                          T* A, size_t lda)
{
    auto p = ce - cs;
    auto stride = (lda - ce) + cs + 1;

    // index starting from left (going right); original matrix index
    auto i = rs*lda + cs + (m0 / p)*lda + (m0 % p);
    auto next_count = m0 % p;

    // index starting from right (going left); original matrix index
    auto j = rs*lda + cs + ((m1 - 1)/p)*lda + ((m1 - 1) % p);
    auto prev_count = (m1 - 1) % p;

    for (auto m = 0; m < l; ++m) {
        using std::swap;
        swap(A[i], A[j]);
        next(&i, &next_count, p, stride);
        prev(&j, &prev_count, p, stride);
    }
}

template <typename T>
void TRIP<T>::reverse_par(size_t m0, size_t m1, size_t l,
                          size_t rs, size_t cs, size_t ce,
                          T* A, size_t lda)
{
    constexpr size_t REVERSE_CUTOFF = 1024;
    if (l <= REVERSE_CUTOFF) {
        reverse_ser(m0, m1, l, rs, cs, ce, A, lda);
    } else {
        auto lm = l / 2;
        tbb::parallel_invoke(
            [=]{ reverse_par(m0, m1, lm, rs, cs, ce, A, lda); },
            [=]{ reverse_par(m0 + lm, m1 - lm, l - lm, rs, cs, ce, A, lda); });
    }
}

template <typename T>
void TRIP<T>::reverse(size_t m0, size_t m1, size_t rs, size_t cs, size_t ce, T* A, size_t lda) {
    reverse_par(m0, m1, (m1 - m0)/2, rs, cs, ce, A, lda);
}

template <typename T>
inline void mitrans(size_t m, size_t n, T* A) {
    TRIP<T>::trip(m, n, A);
}

template <typename T>
inline void mitrans(size_t n, T* A, size_t lda) {
    TRIP<T>::trip(n, A, lda);
}

//==-------------------------------------------------------------------------
// BLAS level-1 (vector-vector) routines
//==-------------------------------------------------------------------------

inline float dot(int n, const float* X, int incX, const float* Y, int incY) {
    return cblas_sdot(n, X, incX, Y, incY);
}

inline double dot(int n, const double* X, int incX, const double* Y, int incY) {
    return cblas_ddot(n, X, incX, Y, incY);
}

inline std::complex<float> dot(int n,
    const std::complex<float>* X, int incX,
    const std::complex<float>* Y, int incY)
{
    std::complex<float> R;
    cblas_cdotu_sub(n, X, incX, Y, incY, &R);
    return R;
}

inline std::complex<double> dot(int n,
    const std::complex<double>* X, int incX,
    const std::complex<double>* Y, int incY)
{
    std::complex<double> R;
    cblas_zdotu_sub(n, X, incX, Y, incY, &R);
    return R;
}

inline float dotc(int n, const float* X, int incX, const float* Y, int incY) {
    return cblas_sdot(n, X, incX, Y, incY);
}

inline float dotc(int n, const double* X, int incX, const double* Y, int incY) {
    return cblas_ddot(n, X, incX, Y, incY);
}

inline std::complex<float> dotc(int n,
    const std::complex<float>* X, int incX,
    const std::complex<float>* Y, int incY)
{
    std::complex<float> R;
    cblas_cdotc_sub(n, X, incX, Y, incY, &R);
    return R;
}

inline std::complex<double> dotc(int n,
    const std::complex<double>* X, int incX,
    const std::complex<double>* Y, int incY)
{
    std::complex<double> R;
    cblas_zdotc_sub(n, X, incX, Y, incY, &R);
    return R;
}

inline float nrm2(int n, const float* X, int incX) {
    return cblas_snrm2(n, X, incX);
}

inline double nrm2(int n, const double* X, int incX) {
    return cblas_dnrm2(n, X, incX);
}

inline std::complex<float> nrm2(int n, const std::complex<float>* X, int incX) {
    return cblas_scnrm2(n, X, incX);
}

inline std::complex<double> nrm2(int n, const std::complex<double>* X, int incX) {
    return cblas_dznrm2(n, X, incX);
}

inline float asum(int n, const float* X, int incX) {
    return cblas_sasum(n, X, incX);
}

inline double asum(int n, const double* X, int incX) {
    return cblas_dasum(n, X, incX);
}

inline std::complex<float> asum(int n, const std::complex<float>* X, int incX) {
    return cblas_scasum(n, X, incX);
}

inline std::complex<double> asum(int n, const std::complex<double>* X, int incX) {
    return cblas_dzasum(n, X, incX);
}

inline int iamax(int n, const float* X, int incX) {
    return cblas_isamax(n, X, incX);
}

inline int iamax(int n, const double* X, int incX) {
    return cblas_idamax(n, X, incX);
}

inline int iamax(int n, const std::complex<float>* X, int incX) {
    return cblas_icamax(n, X, incX);
}

inline int iamax(int n, const std::complex<double>* X, int incX) {
    return cblas_izamax(n, X, incX);
}

inline void swap(int n, float* X, int incX, float* Y, int incY) {
    cblas_sswap(n, X, incX, Y, incY);
}

inline void swap(int n, double* X, int incX, double* Y, int incY) {
    cblas_dswap(n, X, incX, Y, incY);
}

inline void swap(int n, std::complex<float>* X, int incX, std::complex<float>* Y, int incY) {
    cblas_cswap(n, X, incX, Y, incY);
}

inline void swap(int n, std::complex<double>* X, int incX, std::complex<double>* Y, int incY) {
    cblas_zswap(n, X, incX, Y, incY);
}

inline void copy(int n, const float* X, int incX, float* Y, int incY) {
    cblas_scopy(n, X, incX, Y, incY);
}

inline void copy(int n, const double* X, int incX, double* Y, int incY) {
    cblas_dcopy(n, X, incX, Y, incY);
}

inline void copy(int n, const std::complex<float>* X, int incX, std::complex<float>* Y, int incY) {
    cblas_ccopy(n, X, incX, Y, incY);
}

inline void copy(int n, const std::complex<double>* X, int incX, std::complex<double>* Y, int incY) {
    cblas_ccopy(n, X, incX, Y, incY);
}

inline void axpy(int n, float alpha, const float* X, int incX, float* Y, int incY) {
    cblas_saxpy(n, alpha, X, incX, Y, incY);
}

inline void axpy(int n, double alpha, const double* X, int incX, double* Y, int incY) {
    cblas_daxpy(n, alpha, X, incX, Y, incY);
}

inline void axpy(int n, const std::complex<float>& alpha,
                 const std::complex<float>* X, int incX,
                 std::complex<float>* Y, int incY) {
    cblas_caxpy(n, &alpha, X, incX, Y, incY);
}

inline void axpy(int n, const std::complex<double>& alpha,
                 const std::complex<double>* X, int incX,
                 std::complex<double>* Y, int incY) {
    cblas_zaxpy(n, &alpha, X, incX, Y, incY);
}

inline void scal(int n, float a, float* X, int incX) {
    cblas_sscal(n, a, X, incX);
}

inline void scal(int n, double a, double* X, int incX) {
    cblas_dscal(n, a, X, incX);
}

inline void scal(int n, const std::complex<float>& a, std::complex<float>* X, int incX) {
    cblas_cscal(n, &a, X, incX);
}

inline void scal(int n, const std::complex<double>& a, std::complex<double>* X, int incX) {
    cblas_zscal(n, &a, X, incX);
}

//==-------------------------------------------------------------------------
// BLAS level-2 (matrix-vector) routines
//==-------------------------------------------------------------------------

inline void gemv(Layout layout,
                 Transpose transA,
                 int m, int n,
                 float alpha,
                 const float* A, int lda,
                 const float* X, int incX,
                 float beta,
                 float* Y, int incY) {
    cblas_sgemv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                m, n, alpha, A, lda, X, incX, beta, Y, incY);
}

inline void gemv(Layout layout,
                 Transpose transA,
                 int m, int n,
                 double alpha,
                 const double* A, int lda,
                 const double* X, int incX,
                 double beta,
                 double* Y, int incY) {
    cblas_dgemv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                m, n, alpha, A, lda, X, incX, beta, Y, incY);
}

inline void gemv(Layout layout,
                 Transpose transA,
                 int m, int n,
                 const std::complex<float>& alpha,
                 const std::complex<float>* A, int lda,
                 const std::complex<float>* X, int incX,
                 const std::complex<float>& beta,
                 std::complex<float>* Y, int incY) {
    cblas_cgemv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void gemv(Layout layout,
                 Transpose transA,
                 int m, int n,
                 const std::complex<double>& alpha,
                 const std::complex<double>* A, int lda,
                 const std::complex<double>* X, int incX,
                 const std::complex<double>& beta,
                 std::complex<double>* Y, int incY) {
    cblas_zgemv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void symv(Layout layout, Triangle uplo, const int n,
                 float alpha, const float* A, const int lda,
                 const float* x, const int incX,
                 const float beta, float* y, const int incY)
{
    cblas_ssymv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                n, alpha, A, lda, x, incX, beta, y, incY);
}

inline void symv(Layout layout, Triangle uplo, const int n,
                 double alpha, const double* A, const int lda,
                 const double* x, const int incX,
                 const double beta, double* y, const int incY)
{
    cblas_dsymv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                n, alpha, A, lda, x, incX, beta, y, incY);
}

inline void trmv(Layout layout, Triangle uplo, Transpose trans, Diagonal diag,
                 const int n, const float* A, const int lda,
                 float* x, const int incX)
{
    cblas_strmv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                n, A, lda, x, incX);
}

inline void trmv(Layout layout, Triangle uplo, Transpose trans, Diagonal diag,
                 const int n, const double* A, const int lda,
                 double* x, const int incX)
{
    cblas_dtrmv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                n, A, lda, x, incX);
}

inline void trmv(Layout layout, Triangle uplo, Transpose trans, Diagonal diag,
                 const int n, const std::complex<float>* A, const int lda,
                 std::complex<float>* x, const int incX)
{
    cblas_ctrmv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                n, A, lda, x, incX);
}

inline void trmv(Layout layout, Triangle uplo, Transpose trans, Diagonal diag,
                 const int n, const std::complex<double>* A, const int lda,
                 std::complex<double>* x, const int incX)
{
    cblas_ztrmv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                n, A, lda, x, incX);
}

inline void trsv(Layout layout, Triangle uplo, Transpose trans, Diagonal diag,
                 const int n, const float* A, const int lda, float* X, const int incX)
{
    cblas_strsv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasUnit)>(diag),
                n, A, lda, X, incX);
}

inline void trsv(Layout layout, Triangle uplo, Transpose trans, Diagonal diag,
                 const int n, const double* A, const int lda, double* X, const int incX)
{
    cblas_dtrsv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasUnit)>(diag),
                n, A, lda, X, incX);
}

inline void trsv(Layout layout, Triangle uplo, Transpose trans, Diagonal diag,
                 const int n, const std::complex<float>* A, const int lda,
                 std::complex<float>* X, const int incX)
{
    cblas_ctrsv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasUnit)>(diag),
                n, A, lda, X, incX);
}

inline void trsv(Layout layout, Triangle uplo, Transpose trans, Diagonal diag,
                 const int n, const std::complex<double>* A, const int lda,
                 std::complex<double>* X, const int incX)
{
    cblas_ztrsv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasUnit)>(diag),
                n, A, lda, X, incX);
}

inline void ger(Layout layout, const int m, const int n,
                const float alpha,
                const float* X, const int incX,
                const float* Y, const int incY,
                float* A, const int lda)
{
    cblas_sger(static_cast<decltype(CblasRowMajor)>(layout),
               m, n, alpha, X, incX, Y, incY, A, lda);
}

inline void ger(Layout layout, const int m, const int n,
                const double alpha,
                const double* X, const int incX,
                const double* Y, const int incY,
                double* A, const int lda)
{
    cblas_dger(static_cast<decltype(CblasRowMajor)>(layout),
               m, n, alpha, X, incX, Y, incY, A, lda);
}

inline void ger(Layout layout, const int m, const int n,
                const std::complex<float>& alpha,
                const std::complex<float>* X, const int incX,
                const std::complex<float>* Y, const int incY,
                std::complex<float>* A, const int lda)
{
    cblas_cgeru(static_cast<decltype(CblasRowMajor)>(layout),
                m, n, &alpha, X, incX, Y, incY, A, lda);
}

inline void ger(Layout layout, const int m, const int n,
                const std::complex<double>& alpha,
                const std::complex<double>* X, const int incX,
                const std::complex<double>* Y, const int incY,
                std::complex<double>* A, const int lda)
{
    cblas_zgeru(static_cast<decltype(CblasRowMajor)>(layout),
                m, n, &alpha, X, incX, Y, incY, A, lda);
}

inline void gerc(Layout layout, const int m, const int n,
                 const float alpha,
                 const float* X, const int incX,
                 const float* Y, const int incY,
                 float* A, const int lda)
{
    cblas_sger(static_cast<decltype(CblasRowMajor)>(layout),
               m, n, alpha, X, incX, Y, incY, A, lda);
}

inline void gerc(Layout layout, const int m, const int n,
                 const double alpha,
                 const double* X, const int incX,
                 const double* Y, const int incY,
                 double* A, const int lda)
{
    cblas_dger(static_cast<decltype(CblasRowMajor)>(layout),
               m, n, alpha, X, incX, Y, incY, A, lda);
}

inline void gerc(Layout layout, const int m, const int n,
                 const std::complex<float>& alpha,
                 const std::complex<float>* X, const int incX,
                 const std::complex<float>* Y, const int incY,
                 std::complex<float>* A, const int lda)
{
    cblas_cgerc(static_cast<decltype(CblasRowMajor)>(layout),
                m, n, &alpha, X, incX, Y, incY, A, lda);
}

inline void gerc(Layout layout, const int m, const int n,
                 const std::complex<double>& alpha,
                 const std::complex<double>* X, const int incX,
                 const std::complex<double>* Y, const int incY,
                 std::complex<double>* A, const int lda)
{
    cblas_zgerc(static_cast<decltype(CblasRowMajor)>(layout),
                m, n, &alpha, X, incX, Y, incY, A, lda);
}

//==-------------------------------------------------------------------------
// BLAS level-3 (matrix-matrix) routines
//==-------------------------------------------------------------------------

inline void gemm(Layout layout,
                 Transpose transA,
                 Transpose transB,
                 int m, int n, int k,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc) {
    cblas_sgemm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                static_cast<decltype(CblasNoTrans)>(transB),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void gemm(Layout layout,
                 Transpose transA,
                 Transpose transB,
                 int m, int n, int k,
                 double alpha,
                 const double* A, int lda,
                 const double* B, int ldb,
                 double beta,
                 double* C, int ldc) {
    cblas_dgemm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                static_cast<decltype(CblasNoTrans)>(transB),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void gemm(Layout layout,
                 Transpose transA,
                 Transpose transB,
                 int m, int n, int k,
                 const std::complex<float>& alpha,
                 const std::complex<float>* A, int lda,
                 const std::complex<float>* B, int ldb,
                 const std::complex<float> beta,
                 std::complex<float>* C, int ldc) {
    cblas_cgemm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                static_cast<decltype(CblasNoTrans)>(transB),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void gemm(Layout layout,
                 Transpose transA,
                 Transpose transB,
                 int m, int n, int k,
                 const std::complex<double>& alpha,
                 const std::complex<double>* A, int lda,
                 const std::complex<double>* B, int ldb,
                 const std::complex<double> beta,
                 std::complex<double>* C, int ldc) {
    cblas_zgemm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                static_cast<decltype(CblasNoTrans)>(transB),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void symm(Layout layout, Side side, Triangle uplo,
                 const int m, const int n,
                 const float alpha,
                 const float* A, const int lda,
                 const float* B, const int ldb,
                 const float beta,
                 float* C, const int ldc) {
    cblas_ssymm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void symm(Layout layout, Side side, Triangle uplo,
                 const int m, const int n,
                 const double alpha,
                 const double* A, const int lda,
                 const double* B, const int ldb,
                 const double beta,
                 double* C, const int ldc) {
    cblas_dsymm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void symm(Layout layout, Side side, Triangle uplo,
                 const int m, const int n,
                 const std::complex<float> alpha,
                 const std::complex<float>* A, const int lda,
                 const std::complex<float>* B, const int ldb,
                 const std::complex<float> beta,
                 std::complex<float>* C, const int ldc) {
    cblas_csymm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void symm(Layout layout, Side side, Triangle uplo,
                 const int m, const int n,
                 const std::complex<double> alpha,
                 const std::complex<double>* A, const int lda,
                 const std::complex<double>* B, const int ldb,
                 const std::complex<double> beta,
                 std::complex<double>* C, const int ldc) {
    cblas_zsymm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void syrk(Layout layout, Triangle uplo, Transpose trans,
                 const int n, const int k, const float alpha,
                 const float* A, const int lda,
                 const float beta, float* C, const int ldc) {
    cblas_ssyrk(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                n, k, alpha, A, lda, beta, C, ldc);
}

inline void syrk(Layout layout, Triangle uplo, Transpose trans,
                 const int n, const int k, const double alpha,
                 const double* A, const int lda,
                 const double beta, double* C, const int ldc) {
    cblas_dsyrk(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                n, k, alpha, A, lda, beta, C, ldc);
}

inline void syrk(Layout layout, Triangle uplo, Transpose trans,
                 const int n, const int k, const std::complex<float>& alpha,
                 const std::complex<float>* A, const int lda,
                 const std::complex<float>& beta, std::complex<float>* C, const int ldc) {
    cblas_csyrk(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                n, k, &alpha, A, lda, &beta, C, ldc);
}

inline void syrk(Layout layout, Triangle uplo, Transpose trans,
                 const int n, const int k, const std::complex<double>& alpha,
                 const std::complex<double>* A, const int lda,
                 const std::complex<double>& beta, std::complex<double>* C, const int ldc) {
    cblas_csyrk(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                n, k, &alpha, A, lda, &beta, C, ldc);
}

inline void trmm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const int m, const int n, const float alpha,
                 const float* A, const int lda,
                 float* B, const int ldb)
{
    cblas_strmm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                m, n, alpha, A, lda, B, ldb);

}

inline void trmm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const int m, const int n, const double alpha,
                 const double* A, const int lda,
                 double* B, const int ldb)
{
    cblas_dtrmm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                m, n, alpha, A, lda, B, ldb);
}

inline void trmm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const int m, const int n, const std::complex<float>& alpha,
                 const std::complex<float>* A, const int lda,
                 std::complex<float>* B, const int ldb)
{
    cblas_ctrmm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                m, n, &alpha, A, lda, B, ldb);
}

inline void trmm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const int m, const int n, const std::complex<double>& alpha,
                 const std::complex<double>* A, const int lda,
                 std::complex<double>* B, const int ldb)
{
    cblas_ztrmm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                m, n, &alpha, A, lda, B, ldb);
}

inline void trsm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const int m, const int n, const float alpha,
                 const float* A, const int lda, float* B, const int ldb)
{
    cblas_strsm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasUnit)>(diag),
                m, n, alpha, A, lda, B, ldb);
}

inline void trsm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const int m, const int n, const double alpha,
                 const double* A, const int lda, double* B, const int ldb)
{
    cblas_dtrsm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasUnit)>(diag),
                m, n, alpha, A, lda, B, ldb);
}

inline void trsm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const int m, const int n, const std::complex<float>& alpha,
                 const std::complex<float>* A, const int lda,
                 std::complex<float>* B, const int ldb)
{
    cblas_ctrsm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasUnit)>(diag),
                m, n, &alpha, A, lda, B, ldb);
}

inline void trsm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const int m, const int n, const std::complex<double>& alpha,
                 const std::complex<double>* A, const int lda,
                 std::complex<double>* B, const int ldb)
{
    cblas_ctrsm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasUnit)>(diag),
                m, n, &alpha, A, lda, B, ldb);
}

#if HAS_LAPACKE

inline lapack_int getrf(lapack_int m, lapack_int n, float* A, lapack_int lda, lapack_int* ipiv) {
    return LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, A, lda, ipiv);
}

inline lapack_int getrf(lapack_int m, lapack_int n, double* A, lapack_int lda, lapack_int* ipiv) {
    return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, A, lda, ipiv);
}

inline lapack_int getrf(lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, lapack_int* ipiv) {
    return LAPACKE_cgetrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_float*>(A), lda, ipiv);
}

inline lapack_int getrf(lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, lapack_int* ipiv) {
    return LAPACKE_zgetrf(LAPACK_ROW_MAJOR, m, n, reinterpret_cast<lapack_complex_double*>(A), lda, ipiv);
}

inline lapack_int getri(lapack_int n, float* A, lapack_int lda, const lapack_int* ipiv) {
    return LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, A, lda, ipiv);
}

inline lapack_int getri(lapack_int n, double* A, lapack_int lda, const lapack_int* ipiv) {
    return LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, lda, ipiv);
}

inline lapack_int getri(lapack_int n, std::complex<float>* A, lapack_int lda, const lapack_int* ipiv) {
    return LAPACKE_cgetri(LAPACK_ROW_MAJOR, n, reinterpret_cast<lapack_complex_float*>(A), lda, ipiv);
}

inline lapack_int getri(lapack_int n, std::complex<double>* A, lapack_int lda, const lapack_int* ipiv) {
    return LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, reinterpret_cast<lapack_complex_double*>(A), lda, ipiv);
}

inline void getrs(char trans, lapack_int n, lapack_int nrhs,
                  const float* A, lapack_int lda,
                  const lapack_int* ipiv,
                  float* b, lapack_int ldb)
{
    auto info = LAPACKE_sgetrs(LAPACK_ROW_MAJOR, trans, n, nrhs, A, lda, ipiv, b, ldb);
    assert(info == 0);
    (void)info;
}

inline void getrs(char trans, lapack_int n, lapack_int nrhs,
                  const double* A, lapack_int lda,
                  const lapack_int* ipiv,
                  double* b, lapack_int ldb)
{
    auto info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, trans, n, nrhs, A, lda, ipiv, b, ldb);
    assert(info == 0);
    (void)info;
}

inline void getrs(char trans, lapack_int n, lapack_int nrhs,
                  const std::complex<float>* A, lapack_int lda,
                  const lapack_int* ipiv,
                  std::complex<float>* b, lapack_int ldb)
{
    auto info = LAPACKE_cgetrs(
                    LAPACK_ROW_MAJOR, trans, n, nrhs,
                    reinterpret_cast<const lapack_complex_float*>(A), lda,
                    ipiv,
                    reinterpret_cast<lapack_complex_float*>(b), ldb);
    assert(info == 0);
    (void)info;
}

inline void getrs(char trans, lapack_int n, lapack_int nrhs,
                  const std::complex<double>* A, lapack_int lda,
                  const lapack_int* ipiv,
                  std::complex<double>* b, lapack_int ldb)
{
    auto info = LAPACKE_zgetrs(
                    LAPACK_ROW_MAJOR, trans, n, nrhs,
                    reinterpret_cast<const lapack_complex_double*>(A), lda,
                    ipiv,
                    reinterpret_cast<lapack_complex_double*>(b), ldb);
    assert(info == 0);
    (void)info;
}

inline lapack_int potrf(char uplo, lapack_int n, float* A, lapack_int lda) {
    return LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, n, A, lda);
}

inline lapack_int potrf(char uplo, lapack_int n, double* A, lapack_int lda) {
    return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, n, A, lda);
}

inline lapack_int potrf(char uplo, lapack_int n, std::complex<float>* A, lapack_int lda) {
    return LAPACKE_cpotrf(LAPACK_ROW_MAJOR, uplo, n,
                          reinterpret_cast<lapack_complex_float*>(A), lda);
}

inline lapack_int potrf(char uplo, lapack_int n, std::complex<double>* A, lapack_int lda) {
    return LAPACKE_zpotrf(LAPACK_ROW_MAJOR, uplo, n,
                          reinterpret_cast<lapack_complex_double*>(A), lda);
}

inline void gees(char jobvs, char sort,
                 lapack_logical (*select)(const float*, const float*),
                 lapack_int n, float* A, lapack_int lda,
                 lapack_int* sdim, float* wr, float* wi,
                 float* vs, lapack_int ldvs)
{
    auto info = LAPACKE_sgees(LAPACK_ROW_MAJOR, jobvs, sort, select,
                              n, A, lda, sdim, wr, wi, vs, ldvs);

    assert(info >= 0);
    if (info == n + 1)
        throw std::runtime_error("Eigenvalues could not be separated for reordering.");
    if (info == n + 2)
        throw std::runtime_error("Leading eigenvalues do not satisfy sort condition.");
    if (info > 0)
        throw std::runtime_error("Schur form not found.  Possibly ill-conditioned.");
}

inline void gees(char jobvs, char sort,
                 lapack_logical (*select)(const double*, const double*),
                 lapack_int n, double* A, lapack_int lda,
                 lapack_int* sdim, double* wr, double* wi,
                 double* vs, lapack_int ldvs)
{
    auto info = LAPACKE_dgees(LAPACK_ROW_MAJOR, jobvs, sort, select,
                              n, A, lda, sdim, wr, wi, vs, ldvs);

    assert(info >= 0);
    if (info == n + 1)
        throw std::runtime_error("Eigenvalues could not be separated for reordering.");
    if (info == n + 2)
        throw std::runtime_error("Leading eigenvalues do not satisfy sort condition.");
    if (info > 0)
        throw std::runtime_error("Schur form not found.  Possibly ill-conditioned.");
}

inline void gees(char jobvs, char sort,
                 lapack_logical (*select)(const std::complex<float>*),
                 lapack_int n, std::complex<float>* A, lapack_int lda,
                 lapack_int* sdim, std::complex<float>* w,
                 std::complex<float>* vs, lapack_int ldvs)
{
    auto info = LAPACKE_cgees(
        LAPACK_ROW_MAJOR, jobvs, sort,
        reinterpret_cast<lapack_logical(*)(const lapack_complex_float*)>(select),
        n, reinterpret_cast<lapack_complex_float*>(A), lda, sdim,
        reinterpret_cast<lapack_complex_float*>(w),
        reinterpret_cast<lapack_complex_float*>(vs), ldvs);

    assert(info >= 0);
    if (info == n + 1)
        throw std::runtime_error("Eigenvalues could not be separated for reordering.");
    if (info == n + 2)
        throw std::runtime_error("Leading eigenvalues do not satisfy sort condition.");
    if (info > 0)
        throw std::runtime_error("Schur form not found.  Possibly ill-conditioned.");
}

inline void gees(char jobvs, char sort,
                 lapack_logical (*select)(const std::complex<double>*),
                 lapack_int n, std::complex<double>* A, lapack_int lda,
                 lapack_int* sdim, std::complex<double>* w,
                 std::complex<double>* vs, lapack_int ldvs)
{
    auto info = LAPACKE_zgees(
        LAPACK_ROW_MAJOR, jobvs, sort,
        reinterpret_cast<lapack_logical(*)(const lapack_complex_double*)>(select),
        n, reinterpret_cast<lapack_complex_double*>(A), lda, sdim,
        reinterpret_cast<lapack_complex_double*>(w),
        reinterpret_cast<lapack_complex_double*>(vs), ldvs);

    assert(info >= 0);
    if (info == n + 1)
        throw std::runtime_error("Eigenvalues could not be separated for reordering.");
    if (info == n + 2)
        throw std::runtime_error("Leading eigenvalues do not satisfy sort condition.");
    if (info > 0)
        throw std::runtime_error("Schur form not found.  Possibly ill-conditioned.");
}

inline lapack_int gelsd(Layout layout, lapack_int m, lapack_int n, lapack_int nrhs,
                        float* A, lapack_int lda, float* B, lapack_int ldb,
                        float* s, float rcond, lapack_int* rank)
{
    return LAPACKE_sgelsd(layout == Layout::RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR,
                          m, n, nrhs, A, lda, B, ldb, s, rcond, rank);
}

inline lapack_int gelsd(Layout layout, lapack_int m, lapack_int n, lapack_int nrhs,
                        double* A, lapack_int lda, double* B, lapack_int ldb,
                        double* s, double rcond, lapack_int* rank)
{
    return LAPACKE_dgelsd(layout == Layout::RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR,
                          m, n, nrhs, A, lda, B, ldb, s, rcond, rank);
}

inline lapack_int gelsd(Layout layout, lapack_int m, lapack_int n, lapack_int nrhs,
                        std::complex<float>* A, lapack_int lda,
                        std::complex<float>* B, lapack_int ldb,
                        float* s, float rcond, lapack_int* rank)
{
    return LAPACKE_cgelsd(layout == Layout::RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR,
                          m, n, nrhs,
                          reinterpret_cast<lapack_complex_float*>(A), lda,
                          reinterpret_cast<lapack_complex_float*>(B), ldb,
                          s, rcond, rank);
}

inline lapack_int gelsd(Layout layout, lapack_int m, lapack_int n, lapack_int nrhs,
                        std::complex<double>* A, lapack_int lda,
                        std::complex<double>* B, lapack_int ldb,
                        double* s, double rcond, lapack_int* rank)
{
    return LAPACKE_zgelsd(layout == Layout::RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR,
                          m, n, nrhs,
                          reinterpret_cast<lapack_complex_double*>(A), lda,
                          reinterpret_cast<lapack_complex_double*>(B), ldb,
                          s, rcond, rank);
}

#else

inline lapack_int getrf(lapack_int m, lapack_int n, float* A, lapack_int lda, lapack_int* ipiv) {
    lapack_int info;
    sgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

inline lapack_int getrf(lapack_int m, lapack_int n, double* A, lapack_int lda, lapack_int* ipiv) {
    lapack_int info;
    dgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

inline lapack_int getrf(lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, lapack_int* ipiv) {
    lapack_int info;
    cgetrf_(&m, &n, reinterpret_cast<lapack_complex_float*>(A), &lda, ipiv, &info);
    return info;
}

inline lapack_int getrf(lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, lapack_int* ipiv) {
    lapack_int info;
    zgetrf_(&m, &n, reinterpret_cast<lapack_complex_double*>(A), &lda, ipiv, &info);
    return info;
}

inline lapack_int getri(lapack_int n, float* A, lapack_int lda, const lapack_int* ipiv) {
    lapack_int lwork = -1;
    float work_query;
    std::vector<float> work;
    lapack_int info;

    sgetri_(&n, A, &lda, const_cast<lapack_int*>(ipiv), &work_query, &lwork, &info);
    if (info != 0) return info;
    lwork = (lapack_int)work_query;
    work.resize(lwork);
    sgetri_(&n, A, &lda, const_cast<lapack_int*>(ipiv), work.data(), &lwork, &info);
    return info;
}

inline lapack_int getri(lapack_int n, double* A, lapack_int lda, const lapack_int* ipiv) {
    lapack_int lwork = -1;
    double work_query;
    std::vector<double> work;
    lapack_int info;

    dgetri_(&n, A, &lda, const_cast<lapack_int*>(ipiv), &work_query, &lwork, &info);
    if (info != 0) return info;
    lwork = (lapack_int)work_query;
    work.resize(lwork);
    dgetri_(&n, A, &lda, const_cast<lapack_int*>(ipiv), work.data(), &lwork, &info);
    return info;
}

inline lapack_int getri(lapack_int n, std::complex<float>* A, lapack_int lda, const lapack_int* ipiv) {
    lapack_int lwork = -1;
    lapack_complex_float work_query;
    std::vector<lapack_complex_float> work;
    lapack_int info;

    cgetri_(&n, reinterpret_cast<lapack_complex_float*>(A), &lda,
            const_cast<lapack_int*>(ipiv), &work_query, &lwork, &info);
    if (info != 0) return info;
    lwork = (lapack_int)(*((float*)&work_query));
    work.resize(lwork);
    cgetri_(&n, reinterpret_cast<lapack_complex_float*>(A), &lda,
            const_cast<lapack_int*>(ipiv), work.data(), &lwork, &info);
    return info;
}

inline lapack_int getri(lapack_int n, std::complex<double>* A, lapack_int lda, const lapack_int* ipiv) {
    lapack_int lwork = -1;
    lapack_complex_double work_query;
    std::vector<lapack_complex_double> work;
    lapack_int info;

    zgetri_(&n, reinterpret_cast<lapack_complex_double*>(A), &lda,
            const_cast<lapack_int*>(ipiv), &work_query, &lwork, &info);
    if (info != 0) return info;
    lwork = (lapack_int)(*((double*)&work_query));
    work.resize(lwork);
    zgetri_(&n, reinterpret_cast<lapack_complex_double*>(A), &lda,
            const_cast<lapack_int*>(ipiv), work.data(), &lwork, &info);
    return info;
}

inline void getrs(char trans, lapack_int n, lapack_int nrhs,
                  const float* A, lapack_int lda,
                  const lapack_int* ipiv,
                  float* b, lapack_int ldb) {
    lapack_int info;
    sgetrs_(&trans, &n, &nrhs,
            const_cast<float*>(A), &lda,
            const_cast<lapack_int*>(ipiv),
            b, &ldb, &info);
    assert(info == 0);
}

inline void getrs(char trans, lapack_int n, lapack_int nrhs,
                  const double* A, lapack_int lda,
                  const lapack_int* ipiv,
                  double* b, lapack_int ldb) {
    lapack_int info;
    dgetrs_(&trans, &n, &nrhs,
            const_cast<double*>(A), &lda,
            const_cast<lapack_int*>(ipiv),
            b, &ldb, &info);
    assert(info == 0);
}

inline void getrs(char trans, lapack_int n, lapack_int nrhs,
                  const std::complex<float>* A, lapack_int lda,
                  const lapack_int* ipiv,
                  std::complex<float>* b, lapack_int ldb) {
    lapack_int info;
    cgetrs_(&trans, &n, &nrhs,
            const_cast<lapack_complex_float*>(reinterpret_cast<const lapack_complex_float*>(A)), &lda,
            const_cast<lapack_int*>(ipiv),
            reinterpret_cast<lapack_complex_float*>(b), &ldb, &info);
    assert(info == 0);
}

inline void getrs(char trans, lapack_int n, lapack_int nrhs,
                  const std::complex<double>* A, lapack_int lda,
                  const lapack_int* ipiv,
                  std::complex<double>* b, lapack_int ldb) {
    lapack_int info;
    zgetrs_(&trans, &n, &nrhs,
            const_cast<lapack_complex_double*>(reinterpret_cast<const lapack_complex_double*>(A)), &lda,
            const_cast<lapack_int*>(ipiv),
            reinterpret_cast<lapack_complex_double*>(b), &ldb, &info);
    assert(info == 0);
}

inline lapack_int potrf(char uplo, lapack_int n, float* A, lapack_int lda) {
    lapack_int info;
    uplo = uplo == 'L' ? 'U' : 'L';
    spotrf_(&uplo, &n, A, &lda, &info);
    return info;
}

inline lapack_int potrf(char uplo, lapack_int n, double* A, lapack_int lda) {
    lapack_int info;
    uplo = uplo == 'L' ? 'U' : 'L';
    dpotrf_(&uplo, &n, A, &lda, &info);
    return info;
}

inline lapack_int potrf(char uplo, lapack_int n, std::complex<float>* A, lapack_int lda) {
    lapack_int info;
    uplo = uplo == 'L' ? 'U' : 'L';
    cpotrf_(&uplo, &n, reinterpret_cast<lapack_complex_float*>(A), &lda, &info);
    return info;
}

inline lapack_int potrf(char uplo, lapack_int n, std::complex<double>* A, lapack_int lda) {
    lapack_int info;
    uplo = uplo == 'L' ? 'U' : 'L';
    zpotrf_(&uplo, &n, reinterpret_cast<lapack_complex_double*>(A), &lda, &info);
    return info;
}

#if defined(__APPLE__)
#define SELECT_FP(select) reinterpret_cast<__CLPK_L_fp>(select)
#else
#define SELECT_FP(select) (select)
#endif

inline void gees(char jobvs, char sort,
                 lapack_logical (*select)(const float*, const float*),
                 lapack_int n, float* A, lapack_int lda,
                 lapack_int* sdim, float* wr, float* wi,
                 float* vs, lapack_int ldvs)
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    float work_query;
    std::vector<lapack_logical> bwork(n);

    sgees_(&jobvs, &sort, SELECT_FP(select),
           &n, A, &lda, sdim, wr, wi, vs, &ldvs,
           &work_query, &lwork, bwork.data(), &info);

    lwork = static_cast<lapack_int>(work_query);
    std::vector<float> work(lwork);

    mitrans(n, A, lda);
    sgees_(&jobvs, &sort, SELECT_FP(select),
           &n, A, &lda, sdim, wr, wi, vs, &ldvs,
           work.data(), &lwork, bwork.data(), &info);
    mitrans(n, A, lda);
    mitrans(n, vs, ldvs);

    assert(info >= 0);
    if (info == n + 1)
        throw std::runtime_error("Eigenvalues could not be separated for reordering.");
    if (info == n + 2)
        throw std::runtime_error("Leading eigenvalues do not satisfy sort condition.");
    if (info > 0)
        throw std::runtime_error("Schur form not found.  Possibly ill-conditioned.");
}

inline void gees(char jobvs, char sort,
                 lapack_logical (*select)(const double*, const double*),
                 lapack_int n, double* A, lapack_int lda,
                 lapack_int* sdim, double* wr, double* wi,
                 double* vs, lapack_int ldvs)
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    double work_query;
    std::vector<lapack_logical> bwork(n);

    dgees_(&jobvs, &sort, SELECT_FP(select),
           &n, A, &lda, sdim, wr, wi, vs, &ldvs,
           &work_query, &lwork, bwork.data(), &info);

    lwork = static_cast<lapack_int>(work_query);
    std::vector<double> work(lwork);

    mitrans(n, A, lda);
    dgees_(&jobvs, &sort, SELECT_FP(select),
           &n, A, &lda, sdim, wr, wi, vs, &ldvs,
           work.data(), &lwork, bwork.data(), &info);
    mitrans(n, A, lda);
    mitrans(n, vs, ldvs);

    assert(info >= 0);
    if (info == n + 1)
        throw std::runtime_error("Eigenvalues could not be separated for reordering.");
    if (info == n + 2)
        throw std::runtime_error("Leading eigenvalues do not satisfy sort condition.");
    if (info > 0)
        throw std::runtime_error("Schur form not found.  Possibly ill-conditioned.");
}

inline void gees(char jobvs, char sort,
                 lapack_logical (*select)(const std::complex<float>*),
                 lapack_int n, std::complex<float>* A, lapack_int lda,
                 lapack_int* sdim, std::complex<float>* w,
                 std::complex<float>* vs, lapack_int ldvs)
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_float work_query;
    std::vector<float> rwork(n);
    std::vector<lapack_logical> bwork(n);

    cgees_(&jobvs, &sort,
           SELECT_FP(reinterpret_cast<lapack_logical(*)(const lapack_complex_float*)>(select)), &n,
           reinterpret_cast<lapack_complex_float*>(A), &lda, sdim,
           reinterpret_cast<lapack_complex_float*>(w),
           reinterpret_cast<lapack_complex_float*>(vs), &ldvs,
           &work_query, &lwork, rwork.data(), bwork.data(), &info);

    lwork = static_cast<lapack_int>(*reinterpret_cast<float*>(&work_query));
    std::vector<lapack_complex_float> work(lwork);

    mitrans(n, A, lda);
    cgees_(&jobvs, &sort,
           SELECT_FP(reinterpret_cast<lapack_logical(*)(const lapack_complex_float*)>(select)), &n,
           reinterpret_cast<lapack_complex_float*>(A), &lda, sdim,
           reinterpret_cast<lapack_complex_float*>(w),
           reinterpret_cast<lapack_complex_float*>(vs), &ldvs,
           work.data(), &lwork, rwork.data(), bwork.data(), &info);
    mitrans(n, A, lda);
    mitrans(n, vs, ldvs);

    assert(info >= 0);
    if (info == n + 1)
        throw std::runtime_error("Eigenvalues could not be separated for reordering.");
    if (info == n + 2)
        throw std::runtime_error("Leading eigenvalues do not satisfy sort condition.");
    if (info > 0)
        throw std::runtime_error("Schur form not found.  Possibly ill-conditioned.");
}

inline void gees(char jobvs, char sort,
                 lapack_logical (*select)(const std::complex<double>*),
                 lapack_int n, std::complex<double>* A, lapack_int lda,
                 lapack_int* sdim, std::complex<double>* w,
                 std::complex<double>* vs, lapack_int ldvs)
{
    lapack_int info = 0;
    lapack_int lwork = -1;
    lapack_complex_double work_query;
    std::vector<double> rwork(n);
    std::vector<lapack_logical> bwork(n);

    zgees_(&jobvs, &sort,
           SELECT_FP(reinterpret_cast<lapack_logical(*)(const lapack_complex_double*)>(select)), &n,
           reinterpret_cast<lapack_complex_double*>(A), &lda, sdim,
           reinterpret_cast<lapack_complex_double*>(w),
           reinterpret_cast<lapack_complex_double*>(vs), &ldvs,
           &work_query, &lwork, rwork.data(), bwork.data(), &info);

    lwork = static_cast<lapack_int>(*reinterpret_cast<double*>(&work_query));
    std::vector<lapack_complex_double> work(lwork);

    mitrans(n, A, lda);
    zgees_(&jobvs, &sort,
           SELECT_FP(reinterpret_cast<lapack_logical(*)(const lapack_complex_double*)>(select)), &n,
           reinterpret_cast<lapack_complex_double*>(A), &lda, sdim,
           reinterpret_cast<lapack_complex_double*>(w),
           reinterpret_cast<lapack_complex_double*>(vs), &ldvs,
           work.data(), &lwork, rwork.data(), bwork.data(), &info);
    mitrans(n, A, lda);
    mitrans(n, vs, ldvs);

    assert(info >= 0);
    if (info == n + 1)
        throw std::runtime_error("Eigenvalues could not be separated for reordering.");
    if (info == n + 2)
        throw std::runtime_error("Leading eigenvalues do not satisfy sort condition.");
    if (info > 0)
        throw std::runtime_error("Schur form not found.  Possibly ill-conditioned.");
}

inline lapack_int gelsd(Layout layout, lapack_int m, lapack_int n, lapack_int nrhs,
                        float* A, lapack_int lda, float* B, lapack_int ldb,
                        float* s, float rcond, lapack_int* rank)
{
    lapack_int info = 0;

    if (layout == Layout::RowMajor) {
        lapack_int maxmn = std::max(m, n);
        auto A_temp = std::vector<float>(m * n);
        auto B_temp = std::vector<float>(maxmn * nrhs);
        omatcopy(Transpose::Trans, m, n, A, lda, A_temp.data(), m);
        omatcopy(Transpose::Trans, maxmn, nrhs, B, ldb, B_temp.data(), maxmn);
        info = gelsd(Layout::ColMajor, m, n, nrhs,
                     A_temp.data(), m, B_temp.data(), maxmn,
                     s, rcond, rank);
        omatcopy(Transpose::Trans, n, m, A_temp.data(), m, A, lda);
        omatcopy(Transpose::Trans, nrhs, maxmn, B_temp.data(), maxmn, B, ldb);
    } else {
        float work_query;
        lapack_int iwork_query;
        lapack_int lwork = -1;

        sgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, rank,
                &work_query, &lwork, &iwork_query, &info);
        if (info != 0) return info;
        lwork = static_cast<lapack_int>(work_query);

        auto work = std::vector<float>(lwork);
        auto iwork = std::vector<lapack_int>(iwork_query);
        sgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, rank,
                work.data(), &lwork, iwork.data(), &info);
    }
    return info;
}

inline lapack_int gelsd(Layout layout, lapack_int m, lapack_int n, lapack_int nrhs,
                        double* A, lapack_int lda, double* B, lapack_int ldb,
                        double* s, double rcond, lapack_int* rank)
{
    lapack_int info = 0;

    if (layout == Layout::RowMajor) {
        lapack_int maxmn = std::max(m, n);
        auto A_temp = std::vector<double>(m * n);
        auto B_temp = std::vector<double>(maxmn * nrhs);
        omatcopy(Transpose::Trans, m, n, A, lda, A_temp.data(), m);
        omatcopy(Transpose::Trans, maxmn, nrhs, B, ldb, B_temp.data(), maxmn);
        info = gelsd(Layout::ColMajor, m, n, nrhs,
                     A_temp.data(), m, B_temp.data(), maxmn,
                     s, rcond, rank);
        omatcopy(Transpose::Trans, n, m, A_temp.data(), m, A, lda);
        omatcopy(Transpose::Trans, nrhs, maxmn, B_temp.data(), maxmn, B, ldb);
    } else {
        double work_query;
        lapack_int iwork_query;
        lapack_int lwork = -1;

        dgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, rank,
                &work_query, &lwork, &iwork_query, &info);
        if (info != 0) return info;
        lwork = static_cast<lapack_int>(work_query);

        auto work = std::vector<double>(lwork);
        auto iwork = std::vector<lapack_int>(iwork_query);
        dgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, s, &rcond, rank,
                work.data(), &lwork, iwork.data(), &info);
    }
    return info;
}

inline lapack_int gelsd(Layout layout, lapack_int m, lapack_int n, lapack_int nrhs,
                        std::complex<float>* A, lapack_int lda,
                        std::complex<float>* B, lapack_int ldb,
                        float* s, float rcond, lapack_int* rank)
{
    lapack_int info = 0;

    if (layout == Layout::RowMajor) {
        lapack_int maxmn = std::max(m, n);
        auto A_temp = std::vector<std::complex<float>>(m * n);
        auto B_temp = std::vector<std::complex<float>>(maxmn * nrhs);
        omatcopy(Transpose::Trans, m, n, A, lda, A_temp.data(), m);
        omatcopy(Transpose::Trans, maxmn, nrhs, B, ldb, B_temp.data(), maxmn);
        info = gelsd(Layout::ColMajor, m, n, nrhs,
                     A_temp.data(), m, B_temp.data(), maxmn,
                     s, rcond, rank);
        omatcopy(Transpose::Trans, n, m, A_temp.data(), m, A, lda);
        omatcopy(Transpose::Trans, nrhs, maxmn, B_temp.data(), maxmn, B, ldb);
    } else {
        lapack_complex_float work_query;
        float      rwork_query;
        lapack_int iwork_query;
        lapack_int lwork = -1;

        cgelsd_(&m, &n, &nrhs,
                reinterpret_cast<lapack_complex_float*>(A), &lda,
                reinterpret_cast<lapack_complex_float*>(B), &ldb,
                s, &rcond, rank, &work_query, &lwork, &rwork_query, &iwork_query, &info);
        if (info != 0) return info;
        lwork = static_cast<lapack_int>(*reinterpret_cast<float*>(&work_query));

        auto work = std::vector<lapack_complex_float>(lwork);
        auto rwork = std::vector<float>(static_cast<lapack_int>(rwork_query));
        auto iwork = std::vector<lapack_int>(iwork_query);
        cgelsd_(&m, &n, &nrhs,
                reinterpret_cast<lapack_complex_float*>(A), &lda,
                reinterpret_cast<lapack_complex_float*>(B), &ldb,
                s, &rcond, rank, work.data(), &lwork, rwork.data(), iwork.data(), &info);
    }
    return info;
}

inline lapack_int gelsd(Layout layout, lapack_int m, lapack_int n, lapack_int nrhs,
                        std::complex<double>* A, lapack_int lda,
                        std::complex<double>* B, lapack_int ldb,
                        double* s, double rcond, lapack_int* rank)
{
    lapack_int info = 0;

    if (layout == Layout::RowMajor) {
        lapack_int maxmn = std::max(m, n);
        auto A_temp = std::vector<std::complex<double>>(m * n);
        auto B_temp = std::vector<std::complex<double>>(maxmn * nrhs);
        omatcopy(Transpose::Trans, m, n, A, lda, A_temp.data(), m);
        omatcopy(Transpose::Trans, maxmn, nrhs, B, ldb, B_temp.data(), maxmn);
        info = gelsd(Layout::ColMajor, m, n, nrhs,
                     A_temp.data(), m, B_temp.data(), maxmn,
                     s, rcond, rank);
        omatcopy(Transpose::Trans, n, m, A_temp.data(), m, A, lda);
        omatcopy(Transpose::Trans, nrhs, maxmn, B_temp.data(), maxmn, B, ldb);
    } else {
        lapack_complex_double work_query;
        double     rwork_query;
        lapack_int iwork_query;
        lapack_int lwork = -1;

        zgelsd_(&m, &n, &nrhs,
                reinterpret_cast<lapack_complex_double*>(A), &lda,
                reinterpret_cast<lapack_complex_double*>(B), &ldb,
                s, &rcond, rank, &work_query, &lwork, &rwork_query, &iwork_query, &info);
        if (info != 0) return info;
        lwork = static_cast<lapack_int>(*reinterpret_cast<double*>(&work_query));

        auto work = std::vector<lapack_complex_double>(lwork);
        auto rwork = std::vector<double>(static_cast<lapack_int>(rwork_query));
        auto iwork = std::vector<lapack_int>(iwork_query);
        zgelsd_(&m, &n, &nrhs,
                reinterpret_cast<lapack_complex_double*>(A), &lda,
                reinterpret_cast<lapack_complex_double*>(B), &ldb,
                s, &rcond, rank, work.data(), &lwork, rwork.data(), iwork.data(), &info);
    }
    return info;
}

#endif

} // namespace cblas
