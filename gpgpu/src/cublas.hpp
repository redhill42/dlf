#pragma once

#include "gpgpu_cu.hpp"

#if HAS_CUDA

namespace gpgpu { namespace blas {

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

inline void cublasSwapEx(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) {
    ::cublasSswap(handle, n, x, incx, y, incy);
}

inline void cublasSwapEx(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) {
    ::cublasDswap(handle, n, x, incx, y, incy);
}

inline void cublasSwapEx(cublasHandle_t handle, int n, float2* x, int incx, float2* y, int incy) {
    ::cublasCswap(handle, n, reinterpret_cast<cuComplex*>(x), incx, reinterpret_cast<cuComplex*>(y), incy);
}

inline void cublasSwapEx(cublasHandle_t handle, int n, double2* x, int incx, double2* y, int incy) {
    ::cublasZswap(handle, n, reinterpret_cast<cuDoubleComplex*>(x), incx, reinterpret_cast<cuDoubleComplex*>(y), incy);
}

inline void cublasCopyEx(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
    ::cublasScopy(handle, n, x, incx, y, incy);
}

inline void cublasCopyEx(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
    ::cublasDcopy(handle, n, x, incx, y, incy);
}

inline void cublasCopyEx(cublasHandle_t handle, int n, const float2* x, int incx, float2* y, int incy) {
    ::cublasCcopy(handle, n, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<cuComplex*>(y), incy);
}

inline void cublasCopyEx(cublasHandle_t handle, int n, const double2* x, int incx, double2* y, int incy) {
    ::cublasZcopy(handle, n, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<cuDoubleComplex*>(y), incy);
}

inline void cublasAmaxEx(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
    ::cublasIsamax(handle, n, x, incx, result);
}

inline void cublasAmaxEx(cublasHandle_t handle, int n, const double* x, int incx, int* result) {
    ::cublasIdamax(handle, n, x, incx, result);
}

inline void cublasAmaxEx(cublasHandle_t handle, int n, const float2* x, int incx, int* result) {
    ::cublasIcamax(handle, n, reinterpret_cast<const cuComplex*>(x), incx, result);
}

inline void cublasAmaxEx(cublasHandle_t handle, int n, const double2* x, int incx, int* result) {
    ::cublasIzamax(handle, n, reinterpret_cast<const cuDoubleComplex*>(x), incx, result);
}

inline void cublasAminEx(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
    ::cublasIsamin(handle, n, x, incx, result);
}

inline void cublasAminEx(cublasHandle_t handle, int n, const double* x, int incx, int* result) {
    ::cublasIdamin(handle, n, x, incx, result);
}

inline void cublasAminEx(cublasHandle_t handle, int n, const float2* x, int incx, int* result) {
    ::cublasIcamin(handle, n, reinterpret_cast<const cuComplex*>(x), incx, result);
}

inline void cublasAminEx(cublasHandle_t handle, int n, const double2* x, int incx, int* result) {
    ::cublasIzamin(handle, n, reinterpret_cast<const cuDoubleComplex*>(x), incx, result);
}

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

inline void cublasGemvEx(cublasHandle_t handle, cublasOperation_t trans,
                         int m, int n, float alpha,
                         const float* a, int lda, const float* x, int x_inc,
                         float beta, float* y, int y_inc)
{
    ::cublasSgemv(handle, trans, m, n, &alpha, a, lda, x, x_inc, &beta, y, y_inc);
}

inline void cublasGemvEx(cublasHandle_t handle, cublasOperation_t trans,
                         int m, int n, double alpha,
                         const double* a, int lda, const double* x, int x_inc,
                         double beta, double* y, int y_inc)
{
    ::cublasDgemv(handle, trans, m, n, &alpha, a, lda, x, x_inc, &beta, y, y_inc);
}

inline void cublasGemvEx(cublasHandle_t handle, cublasOperation_t trans,
                         int m, int n, float2 alpha,
                         const float2* a, int lda, const float2* x, int x_inc,
                         float2 beta, float2* y, int y_inc)
{
    ::cublasCgemv(handle, trans, m, n,
                  reinterpret_cast<cuComplex*>(&alpha),
                  reinterpret_cast<const cuComplex*>(a), lda,
                  reinterpret_cast<const cuComplex*>(x), x_inc,
                  reinterpret_cast<cuComplex*>(&beta),
                  reinterpret_cast<cuComplex*>(y), y_inc);
}

inline void cublasGemvEx(cublasHandle_t handle, cublasOperation_t trans,
                         int m, int n, double2 alpha,
                         const double2* a, int lda, const double2* x, int x_inc,
                         double2 beta, double2* y, int y_inc)
{
    ::cublasZgemv(handle, trans, m, n,
                  reinterpret_cast<cuDoubleComplex*>(&alpha),
                  reinterpret_cast<const cuDoubleComplex*>(a), lda,
                  reinterpret_cast<const cuDoubleComplex*>(x), x_inc,
                  reinterpret_cast<cuDoubleComplex*>(&beta),
                  reinterpret_cast<cuDoubleComplex*>(y), y_inc);
}

inline void cublasSymv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       const int n, const float alpha,
                       const float* A, const int lda,
                       const float* x, const int incX,
                       const float beta,
                       float* y, const int incY)
{
    ::cublasSsymv(handle, uplo, n, &alpha, A, lda, x, incX, &beta, y, incY);
}

inline void cublasSymv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       const int n, const double alpha,
                       const double* A, const int lda,
                       const double* x, const int incX,
                       const double beta,
                       double* y, const int incY)
{
    ::cublasDsymv(handle, uplo, n, &alpha, A, lda, x, incX, &beta, y, incY);
}

inline void cublasTrmv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       cublasOperation_t trans,
                       cublasDiagType_t diag,
                       int n,
                       const float* A, int lda,
                       float* x, int incx)
{
    ::cublasStrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cublasTrmv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       cublasOperation_t trans,
                       cublasDiagType_t diag,
                       int n,
                       const double* A, int lda,
                       double* x, int incx)
{
    ::cublasDtrmv(handle, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cublasTrmv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       cublasOperation_t trans,
                       cublasDiagType_t diag,
                       int n,
                       const float2* A, int lda,
                       float2* x, int incx)
{
    ::cublasCtrmv(handle, uplo, trans, diag, n,
                  reinterpret_cast<const cuComplex*>(A), lda,
                  reinterpret_cast<cuComplex*>(x), incx);
}

inline void cublasTrmv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       cublasOperation_t trans,
                       cublasDiagType_t diag,
                       int n,
                       const double2* A, int lda,
                       double2* x, int incx)
{
    ::cublasZtrmv(handle, uplo, trans, diag, n,
                  reinterpret_cast<const cuDoubleComplex*>(A), lda,
                  reinterpret_cast<cuDoubleComplex*>(x), incx);
}

inline void cublasTrsv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       cublasOperation_t trans,
                       cublasDiagType_t diag,
                       int n, const float* A, int lda,
                       float* x, int incx)
{
    ::cublasStrsv(handle, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cublasTrsv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       cublasOperation_t trans,
                       cublasDiagType_t diag,
                       int n, const double* A, int lda,
                       double* x, int incx)
{
    ::cublasDtrsv(handle, uplo, trans, diag, n, A, lda, x, incx);
}

inline void cublasTrsv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       cublasOperation_t trans,
                       cublasDiagType_t diag,
                       int n, const float2* A, int lda,
                       float2* x, int incx)
{
    ::cublasCtrsv(handle, uplo, trans, diag, n,
                  reinterpret_cast<const cuComplex*>(A), lda,
                  reinterpret_cast<cuComplex*>(x), incx);
}

inline void cublasTrsv(cublasHandle_t handle,
                       cublasFillMode_t uplo,
                       cublasOperation_t trans,
                       cublasDiagType_t diag,
                       int n, const double2* A, int lda,
                       double2* x, int incx)
{
    ::cublasZtrsv(handle, uplo, trans, diag, n,
                  reinterpret_cast<const cuDoubleComplex*>(A), lda,
                  reinterpret_cast<cuDoubleComplex*>(x), incx);
}

inline void cublasGer(cublasHandle_t handle, const size_t m, const size_t n, float alpha,
                      const float* x, int incx,
                      const float* y, int incy,
                      float* A, int lda)
{
    ::cublasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
}

inline void cublasGer(cublasHandle_t handle, const size_t m, const size_t n, double alpha,
                      const double* x, int incx,
                      const double* y, int incy,
                      double* A, int lda)
{
    ::cublasDger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
}

inline void cublasGer(cublasHandle_t handle, const size_t m, const size_t n, float2 alpha,
                      const float2* x, int incx,
                      const float2* y, int incy,
                      float2* A, int lda)
{
    ::cublasCgeru(handle, m, n,
                  reinterpret_cast<const cuComplex*>(&alpha),
                  reinterpret_cast<const cuComplex*>(x), incx,
                  reinterpret_cast<const cuComplex*>(y), incy,
                  reinterpret_cast<cuComplex*>(A), lda);
}

inline void cublasGer(cublasHandle_t handle, const size_t m, const size_t n, double2 alpha,
                      const double2* x, int incx,
                      const double2* y, int incy,
                      double2* A, int lda)
{
    ::cublasZgeru(handle, m, n,
                  reinterpret_cast<const cuDoubleComplex*>(&alpha),
                  reinterpret_cast<const cuDoubleComplex*>(x), incx,
                  reinterpret_cast<const cuDoubleComplex*>(y), incy,
                  reinterpret_cast<cuDoubleComplex*>(A), lda);
}

inline void cublasGerc(cublasHandle_t handle, const size_t m, const size_t n, float2 alpha,
                       const float2* x, int incx,
                       const float2* y, int incy,
                       float2* A, int lda)
{
    ::cublasCgerc(handle, m, n,
                  reinterpret_cast<const cuComplex*>(&alpha),
                  reinterpret_cast<const cuComplex*>(x), incx,
                  reinterpret_cast<const cuComplex*>(y), incy,
                  reinterpret_cast<cuComplex*>(A), lda);
}

inline void cublasGerc(cublasHandle_t handle, const size_t m, const size_t n, double2 alpha,
                       const double2* x, int incx,
                       const double2* y, int incy,
                       double2* A, int lda)
{
    ::cublasZgerc(handle, m, n,
                  reinterpret_cast<const cuDoubleComplex*>(&alpha),
                  reinterpret_cast<const cuDoubleComplex*>(x), incx,
                  reinterpret_cast<const cuDoubleComplex*>(y), incy,
                  reinterpret_cast<cuDoubleComplex*>(A), lda);
}

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

inline void cublasSymmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         int m, int n,
                         const float* alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         const float* beta,
                         float* C, int ldc)
{
    ::cublasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cublasSymmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         int m, int n,
                         const double* alpha,
                         const double* A, int lda,
                         const double* B, int ldb,
                         const double* beta,
                         double* C, int ldc)
{
    ::cublasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cublasSymmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         int m, int n,
                         const float2* alpha,
                         const float2* A, int lda,
                         const float2* B, int ldb,
                         const float2* beta,
                         float2* C, int ldc)
{
    ::cublasCsymm(handle, side, uplo, m, n,
                  reinterpret_cast<const cuComplex*>(alpha),
                  reinterpret_cast<const cuComplex*>(A), lda,
                  reinterpret_cast<const cuComplex*>(B), ldb,
                  reinterpret_cast<const cuComplex*>(beta),
                  reinterpret_cast<cuComplex*>(C), ldc);
}

inline void cublasSymmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         int m, int n,
                         const double2* alpha,
                         const double2* A, int lda,
                         const double2* B, int ldb,
                         const double2* beta,
                         double2* C, int ldc)
{
    ::cublasZsymm(handle, side, uplo, m, n,
                  reinterpret_cast<const cuDoubleComplex*>(alpha),
                  reinterpret_cast<const cuDoubleComplex*>(A), lda,
                  reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                  reinterpret_cast<const cuDoubleComplex*>(beta),
                  reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

inline void cublasSyrkEx(cublasHandle_t handle,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         int n, int k,
                         const float* alpha, const float* A, int lda,
                         const float* beta, float* C, int ldc)
{
    ::cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

inline void cublasSyrkEx(cublasHandle_t handle,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         int n, int k,
                         const double* alpha, const double* A, int lda,
                         const double* beta, double* C, int ldc)
{
    ::cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

inline void cublasSyrkEx(cublasHandle_t handle,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         int n, int k,
                         const float2* alpha, const float2* A, int lda,
                         const float2* beta, float2* C, int ldc)
{
    ::cublasCsyrk(handle, uplo, trans, n, k,
                  reinterpret_cast<const cuComplex*>(alpha),
                  reinterpret_cast<const cuComplex*>(A), lda,
                  reinterpret_cast<const cuComplex*>(beta),
                  reinterpret_cast<cuComplex*>(C), ldc);
}
inline void cublasSyrkEx(cublasHandle_t handle,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         int n, int k,
                         const double2* alpha, const double2* A, int lda,
                         const double2* beta, double2* C, int ldc)
{
    ::cublasZsyrk(handle, uplo, trans, n, k,
                  reinterpret_cast<const cuDoubleComplex*>(alpha),
                  reinterpret_cast<const cuDoubleComplex*>(A), lda,
                  reinterpret_cast<const cuDoubleComplex*>(beta),
                  reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

inline void cublasTrmmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         cublasDiagType_t diag,
                         int m, int n,
                         const float* alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float* C, int ldc)
{
    ::cublasStrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

inline void cublasTrmmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         cublasDiagType_t diag,
                         int m, int n,
                         const double* alpha,
                         const double* A, int lda,
                         const double* B, int ldb,
                         double* C, int ldc)
{
    ::cublasDtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

inline void cublasTrmmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         cublasDiagType_t diag,
                         int m, int n,
                         const float2* alpha,
                         const float2* A, int lda,
                         const float2* B, int ldb,
                         float2* C, int ldc)
{
    ::cublasCtrmm(handle, side, uplo, trans, diag, m, n,
                  reinterpret_cast<const cuComplex*>(alpha),
                  reinterpret_cast<const cuComplex*>(A), lda,
                  reinterpret_cast<const cuComplex*>(B), ldb,
                  reinterpret_cast<cuComplex*>(C), ldc);
}

inline void cublasTrmmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         cublasDiagType_t diag,
                         int m, int n,
                         const double2* alpha,
                         const double2* A, int lda,
                         const double2* B, int ldb,
                         double2* C, int ldc)
{
    ::cublasZtrmm(handle, side, uplo, trans, diag, m, n,
                  reinterpret_cast<const cuDoubleComplex*>(alpha),
                  reinterpret_cast<const cuDoubleComplex*>(A), lda,
                  reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                  reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

inline void cublasTrsmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         cublasDiagType_t diag,
                         int m, int n,
                         const float* alpha,
                         const float* A, int lda,
                         float* B, int ldb)
{
    ::cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

inline void cublasTrsmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         cublasDiagType_t diag,
                         int m, int n,
                         const double* alpha,
                         const double* A, int lda,
                         double* B, int ldb)
{
    ::cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

inline void cublasTrsmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         cublasDiagType_t diag,
                         int m, int n,
                         const float2* alpha,
                         const float2* A, int lda,
                         float2* B, int ldb)
{
    ::cublasCtrsm(handle, side, uplo, trans, diag, m, n,
                  reinterpret_cast<const cuComplex*>(alpha),
                  reinterpret_cast<const cuComplex*>(A), lda,
                  reinterpret_cast<cuComplex*>(B), ldb);
}

inline void cublasTrsmEx(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t trans,
                         cublasDiagType_t diag,
                         int m, int n,
                         const double2* alpha,
                         const double2* A, int lda,
                         double2* B, int ldb)
{
    ::cublasZtrsm(handle, side, uplo, trans, diag, m, n,
                  reinterpret_cast<const cuDoubleComplex*>(alpha),
                  reinterpret_cast<const cuDoubleComplex*>(A), lda,
                  reinterpret_cast<cuDoubleComplex*>(B), ldb);
}

#if CUBLAS_VER_MAJOR >= 10
template <typename T>
inline void cublasGemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const T* alpha,
    const T* const Aarray[], cudaDataType Atype, int lda,
    const T* const Barray[], cudaDataType Btype, int ldb,
    const T* beta,
    T* const Carray[], cudaDataType Ctype,  int ldc,
    int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo)
{
    ::cublasGemmBatchedEx(
        handle, transa, transb,
        m, n, k,
        reinterpret_cast<const void*>(alpha),
        reinterpret_cast<const void* const*>(Aarray), Atype, lda,
        reinterpret_cast<const void* const*>(Barray), Btype, ldb,
        reinterpret_cast<const void*>(beta),
        reinterpret_cast<void* const*>(Carray), Ctype, ldc,
        batchCount, computeType, algo);
}
#else
inline void cublasGemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* Aarray[], cudaDataType, int lda,
    const float*  Barray[], cudaDataType, int ldb,
    const float* beta,
    float* Carray[], cudaDataType, int ldc,
    int batchCount, cudaDataType, cublasGemmAlgo_t)
{
    ::cublasSgemmBatched(
        handle, transa, transb,
        m, n, k,
        alpha,
        Aarray, lda,
        Barray, ldb,
        beta,
        Carray, ldc,
        batchCount);
}

inline void cublasGemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* Aarray[], cudaDataType, int lda,
    const double* Barray[], cudaDataType, int ldb,
    const double* beta,
    double* Carray[], cudaDataType, int ldc,
    int batchCount, cudaDataType, cublasGemmAlgo_t)
{
    ::cublasDgemmBatched(
        handle, transa, transb,
        m, n, k,
        alpha,
        Aarray, lda,
        Barray, ldb,
        beta,
        Carray, ldc,
        batchCount);
}

inline void cublasGemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float2* alpha,
    const float2* Aarray[], cudaDataType, int lda,
    const float2* Barray[], cudaDataType, int ldb,
    const float2* beta,
    float2* Carray[], cudaDataType, int ldc,
    int batchCount, cudaDataType, cublasGemmAlgo_t)
{
    ::cublasCgemmBatched(
        handle, transa, transb,
        m, n, k,
        reinterpret_cast<const cuComplex*>(alpha),
        reinterpret_cast<const cuComplex**>(Aarray), lda,
        reinterpret_cast<const cuComplex**>(Barray), ldb,
        reinterpret_cast<const cuComplex*>(beta),
        reinterpret_cast<cuComplex**>(Carray), ldc,
        batchCount);
}

inline void cublasGemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double2* alpha,
    const double2* Aarray[], cudaDataType, int lda,
    const double2* Barray[], cudaDataType, int ldb,
    const double2* beta,
    double2* Carray[], cudaDataType, int ldc,
    int batchCount, cudaDataType, cublasGemmAlgo_t)
{
    ::cublasZgemmBatched(
        handle, transa, transb,
        m, n, k,
        reinterpret_cast<const cuDoubleComplex*>(alpha),
        reinterpret_cast<const cuDoubleComplex**>(Aarray), lda,
        reinterpret_cast<const cuDoubleComplex**>(Barray), ldb,
        reinterpret_cast<const cuDoubleComplex*>(beta),
        reinterpret_cast<cuDoubleComplex**>(Carray), ldc,
        batchCount);
}
#endif

#if CUBLAS_VER_MAJOR < 10
inline void cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, cudaDataType, int lda, long long int strideA,
    const float* B, cudaDataType, int ldb, long long int strideB,
    const float* beta,
    float* C, cudaDataType, int ldc, long long int strideC,
    int batchCount, cudaDataType, cublasGemmAlgo_t)
{
    ::cublasSgemmStridedBatched(
        handle, transa, transb,
        m, n, k,
        alpha,
        A, lda, strideA,
        B, ldb, strideB,
        beta,
        C, ldc, strideC,
        batchCount);
}

inline void cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, cudaDataType, int lda, long long int strideA,
    const double* B, cudaDataType, int ldb, long long int strideB,
    const double* beta,
    double* C, cudaDataType, int ldc, long long int strideC,
    int batchCount, cudaDataType, cublasGemmAlgo_t)
{
    ::cublasDgemmStridedBatched(
        handle, transa, transb,
        m, n, k,
        alpha,
        A, lda, strideA,
        B, ldb, strideB,
        beta,
        C, ldc, strideC,
        batchCount);
}

inline void cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float2* alpha,
    const float2* A, cudaDataType, int lda, long long int strideA,
    const float2* B, cudaDataType, int ldb, long long int strideB,
    const float2* beta,
    float2* C, cudaDataType, int ldc, long long int strideC,
    int batchCount, cudaDataType, cublasGemmAlgo_t)
{
    ::cublasCgemmStridedBatched(
        handle, transa, transb,
        m, n, k,
        reinterpret_cast<const cuComplex*>(alpha),
        reinterpret_cast<const cuComplex*>(A), lda, strideA,
        reinterpret_cast<const cuComplex*>(B), ldb, strideB,
        reinterpret_cast<const cuComplex*>(beta),
        reinterpret_cast<cuComplex*>(C), ldc, strideC,
        batchCount);
}

inline void cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double2* alpha,
    const double2* A, cudaDataType, int lda, long long int strideA,
    const double2* B, cudaDataType, int ldb, long long int strideB,
    const double2* beta,
    double2* C, cudaDataType, int ldc, long long int strideC,
    int batchCount, cudaDataType, cublasGemmAlgo_t)
{
    ::cublasZgemmStridedBatched(
        handle, transa, transb,
        m, n, k,
        reinterpret_cast<const cuDoubleComplex*>(alpha),
        reinterpret_cast<const cuDoubleComplex*>(A), lda, strideA,
        reinterpret_cast<const cuDoubleComplex*>(B), ldb, strideB,
        reinterpret_cast<const cuDoubleComplex*>(beta),
        reinterpret_cast<cuDoubleComplex*>(C), ldc, strideC,
        batchCount);
}
#endif

// =================================================================================================
// LAPACK routines
// =================================================================================================

inline cusolverStatus_t cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    return ::cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

inline cusolverStatus_t cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    return ::cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

inline cusolverStatus_t cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle, int m, int n, float2* A, int lda, int* lwork)
{
    return ::cusolverDnCgetrf_bufferSize(
        handle, m, n, reinterpret_cast<cuComplex*>(A), lda, lwork);
}

inline cusolverStatus_t cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle, int m, int n, double2* A, int lda, int* lwork)
{
    return ::cusolverDnZgetrf_bufferSize(
        handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, lwork);
}

inline cusolverStatus_t cusolverDnXgetrf(
    cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* work, int* ipiv, int* info)
{
    return ::cusolverDnSgetrf(handle, m, n, A, lda, work, ipiv, info);
}

inline cusolverStatus_t cusolverDnXgetrf(
    cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* work, int* ipiv, int* info)
{
    return ::cusolverDnDgetrf(handle, m, n, A, lda, work, ipiv, info);
}

inline cusolverStatus_t cusolverDnXgetrf(
    cusolverDnHandle_t handle, int m, int n, float2* A, int lda, float2* work, int* ipiv, int* info)
{
    return ::cusolverDnCgetrf(
        handle, m, n, reinterpret_cast<cuComplex*>(A), lda,
        reinterpret_cast<cuComplex*>(work), ipiv, info);
}

inline cusolverStatus_t cusolverDnXgetrf(
    cusolverDnHandle_t handle, int m, int n, double2* A, int lda, double2* work, int* ipiv, int* info)
{
    return ::cusolverDnZgetrf(
        handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda,
        reinterpret_cast<cuDoubleComplex*>(work), ipiv, info);
}

inline cusolverStatus_t cusolverDnXgetrs(
    cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
    const float* A, int lda, const int* ipiv, float* B, int ldb, int* info)
{
    return ::cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline cusolverStatus_t cusolverDnXgetrs(
    cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
    const double* A, int lda, const int* ipiv, double* B, int ldb, int* info)
{
    return ::cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline cusolverStatus_t cusolverDnXgetrs(
    cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
    const float2* A, int lda, const int* ipiv, float2* B, int ldb, int* info)
{
    return ::cusolverDnCgetrs(
        handle, trans, n, nrhs,
        reinterpret_cast<const cuComplex*>(A), lda, ipiv,
        reinterpret_cast<cuComplex*>(B), ldb, info);
}

inline cusolverStatus_t cusolverDnXgetrs(
    cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
    const double2* A, int lda, const int* ipiv, double2* B, int ldb, int* info)
{
    return ::cusolverDnZgetrs(
        handle, trans, n, nrhs,
        reinterpret_cast<const cuDoubleComplex*>(A), lda, ipiv,
        reinterpret_cast<cuDoubleComplex*>(B), ldb, info);
}

}} // namespace gpgpu::blas

#endif
