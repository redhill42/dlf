#pragma once

#include <complex>

#if HAS_MKL
#include <mkl.h>

#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define cblas_saxpby catlas_saxpby
#define cblas_daxpby catlas_daxpby
#define cblas_caxpby catlas_caxpby
#define cblas_zaxpby catlas_zaxpby

#else
#include <cblas.h>
#endif

namespace cblas {

template <typename T>
constexpr bool RequireBlasType =
    std::is_same<T, float>::value ||
    std::is_same<T, double>::value ||
    std::is_same<T, std::complex<float>>::value ||
    std::is_same<T, std::complex<double>>::value;

enum class Layout {
    RowMajor = CblasRowMajor,
    ColMajor = CblasColMajor
};

enum class Transpose {
    NoTrans = CblasNoTrans,
    Trans = CblasTrans,
    ConjTrans = CblasConjTrans
};

enum class Triangle {
    Upper = CblasUpper,
    Lower = CblasLower,
};

enum class Diagonal {
    NonUnit = CblasNonUnit,
    Unit = CblasUnit
};

enum class Side {
    Left = CblasLeft,
    Right = CblasRight
};

//==-------------------------------------------------------------------------
// BLAS level-1 (vector-vector) routines
//==-------------------------------------------------------------------------

inline float asum(size_t N, const float* X, size_t incX) {
    return cblas_sasum(N, X, incX);
}

inline double asum(size_t N, const double* X, size_t incX) {
    return cblas_dasum(N, X, incX);
}

inline void axpy(size_t N, float alpha, const float* X, size_t incX, float* Y, size_t incY) {
    cblas_saxpy(N, alpha, X, incX, Y, incY);
}

inline void axpy(size_t N, double alpha, const double* X, size_t incX, double* Y, size_t incY) {
    cblas_daxpy(N, alpha, X, incX, Y, incY);
}

inline void axpy(size_t N, const std::complex<float>& alpha, const std::complex<float>* X, size_t incX, std::complex<float>* Y, size_t incY) {
    cblas_caxpy(N, &alpha, X, incX, Y, incY);
}

inline void axpy(size_t N, const std::complex<double>& alpha, const std::complex<double>* X, size_t incX, std::complex<double>* Y, size_t incY) {
    cblas_zaxpy(N, &alpha, X, incX, Y, incY);
}

inline void axpby(size_t N, float alpha, const float* X, size_t incX, float beta, float* Y, size_t incY) {
    cblas_saxpby(N, alpha, X, incX, beta, Y, incY);
}

inline void axpby(size_t N, double alpha, const double* X, size_t incX, double beta, double* Y, size_t incY) {
    cblas_daxpby(N, alpha, X, incX, beta, Y, incY);
}

inline void axpby(size_t N, const std::complex<float>& alpha, const std::complex<float>* X, size_t incX, const std::complex<float>& beta, std::complex<float>* Y, size_t incY) {
    cblas_caxpby(N, &alpha, X, incX, &beta, Y, incY);
}

inline void axpby(size_t N, const std::complex<double>& alpha, const std::complex<double>* X, size_t incX, const std::complex<double>& beta, std::complex<double>* Y, size_t incY) {
    cblas_zaxpby(N, &alpha, X, incX, &beta, Y, incY);
}

inline void copy(size_t N, const float* X, size_t incX, float* Y, size_t incY) {
    cblas_scopy(N, X, incX, Y, incY);
}

inline void copy(size_t N, const double* X, size_t incX, double* Y, size_t incY) {
    cblas_dcopy(N, X, incX, Y, incY);
}

inline void copy(size_t N, const std::complex<float>* X, size_t incX, std::complex<float>* Y, size_t incY) {
    cblas_ccopy(N, X, incX, Y, incY);
}

inline void copy(size_t N, const std::complex<double>* X, size_t incX, std::complex<double>* Y, size_t incY) {
    cblas_ccopy(N, X, incX, Y, incY);
}

inline float dot(size_t N, const float* X, size_t incX, const float* Y, size_t incY) {
    return cblas_sdot(N, X, incX, Y, incY);
}

inline double dot(size_t N, const double* X, size_t incX, const double* Y, size_t incY) {
    return cblas_ddot(N, X, incX, Y, incY);
}

inline std::complex<float> dot(size_t N, const std::complex<float>* X, size_t incX, const std::complex<float>* Y, size_t incY) {
    std::complex<float> R;
    cblas_cdotu_sub(N, X, incX, Y, incY, &R);
    return R;
}

inline std::complex<double> dot(size_t N, const std::complex<double>* X, size_t incX, const std::complex<double>* Y, size_t incY) {
    std::complex<double> R;
    cblas_zdotu_sub(N, X, incX, Y, incY, &R);
    return R;
}

inline float nrm2(size_t N, float* X, size_t incX) {
    return cblas_snrm2(N, X, incX);
}

inline double nrm2(size_t N, double* X, size_t incX) {
    return cblas_dnrm2(N, X, incX);
}

inline void scal(size_t N, float a, float* X, size_t incX) {
    cblas_sscal(N, a, X, incX);
}

inline void scal(size_t N, double a, double* X, size_t incX) {
    cblas_dscal(N, a, X, incX);
}

inline void scal(size_t N, const std::complex<float>& a, std::complex<float>* X, size_t incX) {
    cblas_cscal(N, &a, X, incX);
}

inline void scal(size_t N, const std::complex<double>& a, std::complex<double>* X, size_t incX) {
    cblas_zscal(N, &a, X, incX);
}

inline void swap(size_t N, float* X, size_t incX, float* Y, size_t incY) {
    cblas_sswap(N, X, incX, Y, incY);
}

inline void swap(size_t N, double* X, size_t incX, double* Y, size_t incY) {
    cblas_dswap(N, X, incX, Y, incY);
}

inline void swap(size_t N, std::complex<float>* X, size_t incX, std::complex<float>* Y, size_t incY) {
    cblas_cswap(N, X, incX, Y, incY);
}

inline void swap(size_t N, std::complex<double>* X, size_t incX, std::complex<double>* Y, size_t incY) {
    cblas_zswap(N, X, incX, Y, incY);
}

//==-------------------------------------------------------------------------
// BLAS level-2 (matrix-vector) routines
//==-------------------------------------------------------------------------

inline void gemv(Layout layout,
                 Transpose transA,
                 size_t m, size_t n,
                 float alpha,
                 const float* A, size_t lda,
                 const float* X, size_t incX,
                 float beta,
                 float* Y, size_t incY) {
    cblas_sgemv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                m, n, alpha, A, lda, X, incX, beta, Y, incY);
}

inline void gemv(Layout layout,
                 Transpose transA,
                 size_t m, size_t n,
                 double alpha,
                 const double* A, size_t lda,
                 const double* X, size_t incX,
                 double beta,
                 double* Y, size_t incY) {
    cblas_dgemv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                m, n, alpha, A, lda, X, incX, beta, Y, incY);
}

inline void gemv(Layout layout,
                 Transpose transA,
                 size_t m, size_t n,
                 const std::complex<float>& alpha,
                 const std::complex<float>* A, size_t lda,
                 const std::complex<float>* X, size_t incX,
                 const std::complex<float>& beta,
                 std::complex<float>* Y, size_t incY) {
    cblas_cgemv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void gemv(Layout layout,
                 Transpose transA,
                 size_t m, size_t n,
                 const std::complex<double>& alpha,
                 const std::complex<double>* A, size_t lda,
                 const std::complex<double>* X, size_t incX,
                 const std::complex<double>& beta,
                 std::complex<double>* Y, size_t incY) {
    cblas_zgemv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
}

//==-------------------------------------------------------------------------
// BLAS level-3 (matrix-matrix) routines
//==-------------------------------------------------------------------------

inline void gemm(Layout layout,
                 Transpose transA,
                 Transpose transB,
                 size_t m, size_t n, size_t k,
                 float alpha,
                 const float* A, size_t lda,
                 const float* B, size_t ldb,
                 float beta,
                 float* C, size_t ldc) {
    cblas_sgemm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                static_cast<decltype(CblasNoTrans)>(transB),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void gemm(Layout layout,
                 Transpose transA,
                 Transpose transB,
                 size_t m, size_t n, size_t k,
                 double alpha,
                 const double* A, size_t lda,
                 const double* B, size_t ldb,
                 double beta,
                 double* C, size_t ldc) {
    cblas_dgemm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                static_cast<decltype(CblasNoTrans)>(transB),
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void gemm(Layout layout,
                 Transpose transA,
                 Transpose transB,
                 size_t m, size_t n, size_t k,
                 const std::complex<float>& alpha,
                 const std::complex<float>* A, size_t lda,
                 const std::complex<float>* B, size_t ldb,
                 const std::complex<float> beta,
                 std::complex<float>* C, size_t ldc) {
    cblas_cgemm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                static_cast<decltype(CblasNoTrans)>(transB),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void gemm(Layout layout,
                 Transpose transA,
                 Transpose transB,
                 size_t m, size_t n, size_t k,
                 const std::complex<double>& alpha,
                 const std::complex<double>* A, size_t lda,
                 const std::complex<double>* B, size_t ldb,
                 const std::complex<double> beta,
                 std::complex<double>* C, size_t ldc) {
    cblas_zgemm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                static_cast<decltype(CblasNoTrans)>(transB),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void symm(Layout layout, Side side, Triangle uplo,
                 const size_t m, const size_t n,
                 const float alpha,
                 const float* A, const size_t lda,
                 const float* B, const size_t ldb,
                 const float beta,
                 float* C, const size_t ldc) {
    cblas_ssymm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void symm(Layout layout, Side side, Triangle uplo,
                 const size_t m, const size_t n,
                 const double alpha,
                 const double* A, const size_t lda,
                 const double* B, const size_t ldb,
                 const double beta,
                 double* C, const size_t ldc) {
    cblas_dsymm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void symm(Layout layout, Side side, Triangle uplo,
                 const size_t m, const size_t n,
                 const std::complex<float> alpha,
                 const std::complex<float>* A, const size_t lda,
                 const std::complex<float>* B, const size_t ldb,
                 const std::complex<float> beta,
                 std::complex<float>* C, const size_t ldc) {
    cblas_csymm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void symm(Layout layout, Side side, Triangle uplo,
                 const size_t m, const size_t n,
                 const std::complex<double> alpha,
                 const std::complex<double>* A, const size_t lda,
                 const std::complex<double>* B, const size_t ldb,
                 const std::complex<double> beta,
                 std::complex<double>* C, const size_t ldc) {
    cblas_zsymm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void trmm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const size_t m, const size_t n, const float alpha,
                 const float* A, const size_t lda,
                 float* B, const size_t ldb)
{
    cblas_strmm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                m, n, alpha, A, lda, B, ldb);

}

inline void trmm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const size_t m, const size_t n, const double alpha,
                 const double* A, const size_t lda,
                 double* B, const size_t ldb)
{
    cblas_dtrmm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                m, n, alpha, A, lda, B, ldb);

}

inline void trmm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const size_t m, const size_t n, const std::complex<float>& alpha,
                 const std::complex<float>* A, const size_t lda,
                 std::complex<float>* B, const size_t ldb)
{
    cblas_ctrmm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                m, n, &alpha, A, lda, B, ldb);

}

inline void trmm(Layout layout, Side side, Triangle uplo, Transpose trans, Diagonal diag,
                 const size_t m, const size_t n, const std::complex<double>& alpha,
                 const std::complex<double>* A, const size_t lda,
                 std::complex<double>* B, const size_t ldb)
{
    cblas_ztrmm(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasLeft)>(side),
                static_cast<decltype(CblasLower)>(uplo),
                static_cast<decltype(CblasNoTrans)>(trans),
                static_cast<decltype(CblasNonUnit)>(diag),
                m, n, &alpha, A, lda, B, ldb);

}

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
    mkl_cimatcopy(ordering, trans, rows, cols,
                  *reinterpret_cast<const MKL_Complex8*>(&alpha),
                  reinterpret_cast<MKL_Complex8*>(AB),
                  lda, ldb);
    return true;
}

inline bool imatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const std::complex<double>& alpha,
    std::complex<double>* AB, size_t lda, size_t ldb)
{
    mkl_zimatcopy(ordering, trans, rows, cols,
                  *reinterpret_cast<const MKL_Complex16*>(&alpha),
                  reinterpret_cast<MKL_Complex16*>(AB),
                  lda, ldb);
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
    mkl_comatcopy(ordering, trans, rows, cols,
                  *reinterpret_cast<const MKL_Complex8*>(&alpha),
                  reinterpret_cast<const MKL_Complex8*>(A), lda,
                  reinterpret_cast<MKL_Complex8*>(B), ldb);
    return true;
}

inline bool omatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const std::complex<double>& alpha,
    const std::complex<double>* A, size_t lda,
    std::complex<double>* B, size_t ldb)
{
    mkl_zomatcopy(ordering, trans, rows, cols,
                  *reinterpret_cast<const MKL_Complex16*>(&alpha),
                  reinterpret_cast<const MKL_Complex16*>(A), lda,
                  reinterpret_cast<MKL_Complex16*>(B), ldb);
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
    mkl_comatcopy2(
        ordering, trans, rows, cols,
        *reinterpret_cast<const MKL_Complex8*>(&alpha),
        reinterpret_cast<const MKL_Complex8*>(A), lda, stridea,
        reinterpret_cast<MKL_Complex8*>(B), ldb, strideb);
    return true;
}

inline bool omatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const std::complex<double>& alpha,
    const std::complex<double>* A, size_t lda, size_t stridea,
    std::complex<double>* B, size_t ldb, size_t strideb)
{
    mkl_zomatcopy2(
        ordering, trans, rows, cols,
        *reinterpret_cast<const MKL_Complex16*>(&alpha),
        reinterpret_cast<const MKL_Complex16*>(A), lda, stridea,
        reinterpret_cast<MKL_Complex16*>(B), ldb, strideb);
    return true;
}

#endif //!HAS_MKL

} // namespace cblas
