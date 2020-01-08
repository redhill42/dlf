#pragma once

#include <complex>

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

//==-------------------------------------------------------------------------
// MKL extension
//==-------------------------------------------------------------------------

template <typename T>
inline bool imatcopy(char, char, size_t, size_t, const T&, T*, size_t, size_t) {
    return false;
}

template <typename T>
inline bool omatcopy(char, char, size_t, size_t, const T&, const T*, size_t, T*, size_t) {
    return false;
}

template <typename T>
inline bool omatcopy2(char, char, size_t, size_t, const T&, const T*, size_t, size_t, T*, size_t, size_t) {
    return false;
}

#if HAS_MKL
inline bool imatcopy(
    char ordering, char trans,
    size_t rows, size_t cols, float alpha,
    float* AB, size_t lda, size_t ldb)
{
    mkl_simatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
    return true;
}

inline bool imatcopy(
    char ordering, char trans,
    size_t rows, size_t cols, double alpha,
    double* AB, size_t lda, size_t ldb)
{
    mkl_dimatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
    return true;
}

inline bool imatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const std::complex<float>& alpha,
    std::complex<float>* AB, size_t lda, size_t ldb)
{
    mkl_cimatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
    return true;
}

inline bool imatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const std::complex<double>& alpha,
    std::complex<double>* AB, size_t lda, size_t ldb)
{
    mkl_zimatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
    return true;
}

inline bool omatcopy(
    char ordering, char trans,
    size_t rows, size_t cols, float alpha,
    const float* A, size_t lda,
    float* B, size_t ldb)
{
    mkl_somatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
    return true;
}

inline bool omatcopy(
    char ordering, char trans,
    size_t rows, size_t cols, double alpha,
    const double* A, size_t lda,
    double* B, size_t ldb)
{
    mkl_domatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
    return true;
}

inline bool omatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const std::complex<float>& alpha,
    const std::complex<float>* A, size_t lda,
    std::complex<float>* B, size_t ldb)
{
    mkl_comatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
    return true;
}

inline bool omatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const std::complex<double>& alpha,
    const std::complex<double>* A, size_t lda,
    std::complex<double>* B, size_t ldb)
{
    mkl_zomatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
    return true;
}

inline bool omatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols, float alpha,
    const float* A, size_t lda, size_t stridea,
    float* B, size_t ldb, size_t strideb)
{
    mkl_somatcopy2(ordering, trans, rows, cols, alpha, A, lda, stridea, B, ldb, strideb);
    return true;
}

inline bool omatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols, double alpha,
    const double* A, size_t lda, size_t stridea,
    double* B, size_t ldb, size_t strideb)
{
    mkl_domatcopy2(ordering, trans, rows, cols, alpha, A, lda, stridea, B, ldb, strideb);
    return true;
}

inline bool omatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const std::complex<float>& alpha,
    const std::complex<float>* A, size_t lda, size_t stridea,
    std::complex<float>* B, size_t ldb, size_t strideb)
{
    mkl_comatcopy2(ordering, trans, rows, cols, alpha, A, lda, stridea, B, ldb, strideb);
    return true;
}

inline bool omatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const std::complex<double>& alpha,
    const std::complex<double>* A, size_t lda, size_t stridea,
    std::complex<double>* B, size_t ldb, size_t strideb)
{
    mkl_zomatcopy2(ordering, trans, rows, cols, alpha, A, lda, stridea, B, ldb, strideb);
    return true;
}
#endif //!HAS_MKL

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

#endif

} // namespace cblas
