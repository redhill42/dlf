#ifndef KNERON_OS_BLAS_H
#define KNERON_OS_BLAS_H

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
constexpr bool IsBlasType =
    std::disjunction_v<
        std::is_same<T, float>,
        std::is_same<T, double>,
        std::is_same<T, std::complex<float>>,
        std::is_same<T, std::complex<double>>>;

enum class Layout {
    RowMajor = CblasRowMajor,
    ColMajor = CblasColMajor
};

enum class Transpose {
    NoTrans = CblasNoTrans,
    Trans = CblasTrans,
    ConjTrans = CblasConjTrans
};

inline void axpy(size_t N, float alpha, const float* X, int incX, float* Y, int incY) {
    cblas_saxpy(N, alpha, X, incX, Y, incY);
}

inline void axpy(size_t N, double alpha, const double* X, int incX, double* Y, int incY) {
    cblas_daxpy(N, alpha, X, incX, Y, incY);
}

inline void axpy(size_t N, const std::complex<float>& alpha, const std::complex<float>* X, int incX, std::complex<float>* Y, int incY) {
    cblas_caxpy(N, &alpha, X, incX, Y, incY);
}

inline void axpy(size_t N, const std::complex<double>& alpha, const std::complex<double>* X, int incX, std::complex<double>* Y, int incY) {
    cblas_zaxpy(N, &alpha, X, incX, Y, incY);
}

inline void axpby(size_t N, float alpha, const float* X, int incX, float beta, float* Y, int incY) {
    cblas_saxpby(N, alpha, X, incX, beta, Y, incY);
}

inline void axpby(size_t N, double alpha, const double* X, int incX, double beta, double* Y, int incY) {
    cblas_daxpby(N, alpha, X, incX, beta, Y, incY);
}

inline void axpby(size_t N, const std::complex<float>& alpha, const std::complex<float>* X, int incX, const std::complex<float>& beta, std::complex<float>* Y, int incY) {
    cblas_caxpby(N, &alpha, X, incX, &beta, Y, incY);
}

inline void axpby(size_t N, const std::complex<double>& alpha, const std::complex<double>* X, int incX, const std::complex<double>& beta, std::complex<double>* Y, int incY) {
    cblas_zaxpby(N, &alpha, X, incX, &beta, Y, incY);
}

inline void scal(size_t N, float a, float* X, int incX) {
    cblas_sscal(N, a, X, incX);
}

inline void scal(size_t N, double a, double* X, int incX) {
    cblas_dscal(N, a, X, incX);
}

inline void scal(size_t N, const std::complex<float>& a, std::complex<float>* X, int incX) {
    cblas_cscal(N, &a, X, incX);
}

inline void scal(size_t N, const std::complex<double>& a, std::complex<double>* X, int incX) {
    cblas_zscal(N, &a, X, incX);
}

inline float dot(size_t N, const float* X, int incX, const float* Y, int incY) {
    return cblas_sdot(N, X, incX, Y, incY);
}

inline double dot(size_t N, const double* X, int incX, const double* Y, int incY) {
    return cblas_ddot(N, X, incX, Y, incY);
}

inline void gemv(Layout layout,
                 Transpose transA,
                 size_t m, size_t n,
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
                 size_t m, size_t n,
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
                 size_t m, size_t n,
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
                 size_t m, size_t n,
                 const std::complex<double>& alpha,
                 const std::complex<double>* A, int lda,
                 const std::complex<double>* X, int incX,
                 const std::complex<double>& beta,
                 std::complex<double>* Y, int incY) {
    cblas_zgemv(static_cast<decltype(CblasRowMajor)>(layout),
                static_cast<decltype(CblasNoTrans)>(transA),
                m, n, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void gemm(Layout layout,
                 Transpose transA,
                 Transpose transB,
                 size_t m, size_t n, size_t k,
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
                 size_t m, size_t n, size_t k,
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
                 size_t m, size_t n, size_t k,
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
                 size_t m, size_t n, size_t k,
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

} // namespace cblas

#if HAS_MKL
namespace mkl {

inline void imatcopy(char ordering, char trans, size_t rows, size_t cols, float alpha, float* AB, size_t lda, size_t ldb) {
    mkl_simatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
}

inline void imatcopy(char ordering, char trans, size_t rows, size_t cols, double alpha, double* AB, size_t lda, size_t ldb) {
    mkl_dimatcopy(ordering, trans, rows, cols, alpha, AB, lda, ldb);
}

inline void imatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<float>& alpha, std::complex<float>* AB, size_t lda, size_t ldb) {
    mkl_cimatcopy(ordering, trans, rows, cols,
                  *reinterpret_cast<const MKL_Complex8*>(&alpha),
                  reinterpret_cast<MKL_Complex8*>(AB),
                  lda, ldb);
}

inline void imatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<double>& alpha, std::complex<double>* AB, size_t lda, size_t ldb) {
    mkl_zimatcopy(ordering, trans, rows, cols,
                  *reinterpret_cast<const MKL_Complex16*>(&alpha),
                  reinterpret_cast<MKL_Complex16*>(AB),
                  lda, ldb);
}

inline void omatcopy(char ordering, char trans, size_t rows, size_t cols, float alpha, const float* A, size_t lda, float* B, size_t ldb) {
    mkl_somatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
}

inline void omatcopy(char ordering, char trans, size_t rows, size_t cols, double alpha, const double* A, size_t lda, double* B, size_t ldb) {
    mkl_domatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);
}

inline void omatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<float>& alpha, const std::complex<float>* A, size_t lda, std::complex<float>* B, size_t ldb) {
    mkl_comatcopy(ordering, trans, rows, cols,
                  *reinterpret_cast<const MKL_Complex8*>(&alpha),
                  reinterpret_cast<const MKL_Complex8*>(A), lda,
                  reinterpret_cast<MKL_Complex8*>(B), ldb);
}

inline void omatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<double>& alpha, const std::complex<double>* A, size_t lda, std::complex<double>* B, size_t ldb) {
    mkl_zomatcopy(ordering, trans, rows, cols,
                  *reinterpret_cast<const MKL_Complex16*>(&alpha),
                  reinterpret_cast<const MKL_Complex16*>(A), lda,
                  reinterpret_cast<MKL_Complex16*>(B), ldb);
}

} // namespace mkl
#endif // HAS_MKL

#endif //KNERON_OS_BLAS_H
