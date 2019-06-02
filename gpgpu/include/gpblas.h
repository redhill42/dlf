
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the interface to the CLBlast BLAS routines. It also contains the definitions
// of the returned status codes and the layout and transpose types. This is the only header users
// of CLBlast should include and use.
//
// =================================================================================================

#ifndef GPGPU_BLAS_H_
#define GPGPU_BLAS_H_

#include <cstdlib> // For size_t
#include <string> // For OverrideParameters function
#include <unordered_map> // For OverrideParameters function

#include "gpgpu.h"

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#if defined(_WIN32) && defined(GPGPU_DLL)
  #if defined(COMPILING_DLL)
    #define PUBLIC_API __declspec(dllexport)
  #else
    #define PUBLIC_API __declspec(dllimport)
  #endif
#else
  #define PUBLIC_API
#endif

namespace gpgpu { namespace blas {
// =================================================================================================

// Status codes. These codes can be returned by functions declared in this header file. The error
// codes match either the standard OpenCL error codes or the clBLAS error codes. 
enum class StatusCode {
    // Status codes in common with the OpenCL standard
    kSuccess                   =   0, // CL_SUCCESS
    kOpenCLCompilerNotAvailable=  -3, // CL_COMPILER_NOT_AVAILABLE
    kTempBufferAllocFailure    =  -4, // CL_MEM_OBJECT_ALLOCATION_FAILURE
    kOpenCLOutOfResources      =  -5, // CL_OUT_OF_RESOURCES
    kOpenCLOutOfHostMemory     =  -6, // CL_OUT_OF_HOST_MEMORY
    kOpenCLBuildProgramFailure = -11, // CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
    kInvalidValue              = -30, // CL_INVALID_VALUE
    kInvalidCommandQueue       = -36, // CL_INVALID_COMMAND_QUEUE
    kInvalidMemObject          = -38, // CL_INVALID_MEM_OBJECT
    kInvalidBinary             = -42, // CL_INVALID_BINARY
    kInvalidBuildOptions       = -43, // CL_INVALID_BUILD_OPTIONS
    kInvalidProgram            = -44, // CL_INVALID_PROGRAM
    kInvalidProgramExecutable  = -45, // CL_INVALID_PROGRAM_EXECUTABLE
    kInvalidKernelName         = -46, // CL_INVALID_KERNEL_NAME
    kInvalidKernelDefinition   = -47, // CL_INVALID_KERNEL_DEFINITION
    kInvalidKernel             = -48, // CL_INVALID_KERNEL
    kInvalidArgIndex           = -49, // CL_INVALID_ARG_INDEX
    kInvalidArgValue           = -50, // CL_INVALID_ARG_VALUE
    kInvalidArgSize            = -51, // CL_INVALID_ARG_SIZE
    kInvalidKernelArgs         = -52, // CL_INVALID_KERNEL_ARGS
    kInvalidLocalNumDimensions = -53, // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
    kInvalidLocalThreadsTotal  = -54, // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
    kInvalidLocalThreadsDim    = -55, // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
    kInvalidGlobalOffset       = -56, // CL_INVALID_GLOBAL_OFFSET
    kInvalidEventWaitList      = -57, // CL_INVALID_EVENT_WAIT_LIST
    kInvalidEvent              = -58, // CL_INVALID_EVENT
    kInvalidOperation          = -59, // CL_INVALID_OPERATION
    kInvalidBufferSize         = -61, // CL_INVALID_BUFFER_SIZE
    kInvalidGlobalWorkSize     = -63, // CL_INVALID_GLOBAL_WORK_SIZE

    // Status codes in common with the clBLAS library
    kNotImplemented            = -1024, // Routine or functionality not implemented yet
    kInvalidMatrixA            = -1022, // Matrix A is not a valid OpenCL buffer
    kInvalidMatrixB            = -1021, // Matrix B is not a valid OpenCL buffer
    kInvalidMatrixC            = -1020, // Matrix C is not a valid OpenCL buffer
    kInvalidVectorX            = -1019, // Vector X is not a valid OpenCL buffer
    kInvalidVectorY            = -1018, // Vector Y is not a valid OpenCL buffer
    kInvalidDimension          = -1017, // Dimensions M, N, and K have to be larger than zero
    kInvalidLeadDimA           = -1016, // LD of A is smaller than the matrix's first dimension
    kInvalidLeadDimB           = -1015, // LD of B is smaller than the matrix's first dimension
    kInvalidLeadDimC           = -1014, // LD of C is smaller than the matrix's first dimension
    kInvalidIncrementX         = -1013, // Increment of vector X cannot be zero
    kInvalidIncrementY         = -1012, // Increment of vector Y cannot be zero
    kInsufficientMemoryA       = -1011, // Matrix A's OpenCL buffer is too small
    kInsufficientMemoryB       = -1010, // Matrix B's OpenCL buffer is too small
    kInsufficientMemoryC       = -1009, // Matrix C's OpenCL buffer is too small
    kInsufficientMemoryX       = -1008, // Vector X's OpenCL buffer is too small
    kInsufficientMemoryY       = -1007, // Vector Y's OpenCL buffer is too small

    // Custom additional status codes for CLBlast
    kInsufficientMemoryTemp    = -2050, // Temporary buffer provided to GEMM routine is too small
    kInvalidBatchCount         = -2049, // The batch count needs to be positive
    kInvalidOverrideKernel     = -2048, // Trying to override parameters for an invalid kernel
    kMissingOverrideParameter  = -2047, // Missing override parameter(s) for the target kernel
    kInvalidLocalMemUsage      = -2046, // Not enough local memory available on this device
    kNoHalfPrecision           = -2045, // Half precision (16-bits) not supported by the device
    kNoDoublePrecision         = -2044, // Double precision (64-bits) not supported by the device
    kInvalidVectorScalar       = -2043, // The unit-sized vector is not a valid OpenCL buffer
    kInsufficientMemoryScalar  = -2042, // The unit-sized vector's OpenCL buffer is too small
    kDatabaseError             = -2041, // Entry for the device was not found in the database
    kUnknownError              = -2040, // A catch-all error code representing an unspecified error
    kUnexpectedError           = -2039, // A catch-all error code representing an unexpected exception
};

// Matrix layout and transpose types
enum class Layout     { RowMajor = 101, ColMajor = 102 };
enum class Transpose  { NoTrans = 111, Trans = 112, ConjTrans = 113 };
enum class Triangle   { Upper = 121, Lower = 122 };
enum class Diagonal   { NonUnit = 131, Unit = 132 };
enum class Side       { Left = 141, Right = 142 };
enum class KernelMode { CrossCorrelation = 151, Convolution = 152 };

// Precision scoped enum (values in bits)
enum class Precision  { Half = 16, Single = 32, Double = 64,
                        ComplexSingle = 3232, ComplexDouble = 6464, Any = -1 };

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
template <typename T>
void rotg(Buffer<T>& sa_buffer, const size_t sa_offset,
          Buffer<T>& sb_buffer, const size_t sb_offset,
          Buffer<T>& sc_buffer, const size_t sc_offset,
          Buffer<T>& ss_buffer, const size_t ss_offset,
          const Queue& queue, Event* event = nullptr);

// Generate modified givens plane rotation: SROTMG/DROTMG
template <typename T>
void rotmg(Buffer<T>& sd1_buffer, const size_t sd1_offset,
           Buffer<T>& sd2_buffer, const size_t sd2_offset,
           Buffer<T>& sx1_buffer, const size_t sx1_offset,
           const Buffer<T>& sy1_buffer, const size_t sy1_offset,
           Buffer<T>& sparam_buffer, const size_t sparam_offset,
           const Queue& queue, Event* event = nullptr);

// Apply givens plane rotation: SROT/DROT
template <typename T>
void rot(const size_t n,
         Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         const T cos,
         const T sin,
         const Queue& queue, Event* event = nullptr);

// Apply modified givens plane rotation: SROTM/DROTM
template <typename T>
void rotm(const size_t n,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& sparam_buffer, const size_t sparam_offset,
          const Queue& queue, Event* event = nullptr);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
template <typename T>
void swap(const size_t n,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
template <typename T>
void scal(const size_t n,
          const T alpha,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event = nullptr);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
template <typename T>
void copy(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
template <typename T>
void axpy(const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Dot product of two vectors: SDOT/DDOT/HDOT
template <typename T>
void dot(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& dot_buffer, const size_t dot_offset,
         const Queue& queue, Event* event = nullptr);

// Dot product of two complex vectors: CDOTU/ZDOTU
template <typename T>
void dotu(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& dot_buffer, const size_t dot_offset,
          const Queue& queue, Event* event = nullptr);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
template <typename T>
void dotc(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& dot_buffer, const size_t dot_offset,
          const Queue& queue, Event* event = nullptr);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
template <typename T>
void nrm2(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& nrm2_buffer, const size_t nrm2_offset,
          const Queue& queue, Event* event = nullptr);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
template <typename T>
void asum(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& asum_buffer, const size_t asum_offset,
          const Queue& queue, Event* event = nullptr);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
template <typename T>
void sum(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& sum_buffer, const size_t sum_offset,
         const Queue& queue, Event* event = nullptr);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
template <typename T>
void amax(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
          const Queue& queue, Event* event = nullptr);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
template <typename T>
void amin(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<unsigned int>& imin_buffer, const size_t imin_offset,
          const Queue& queue, Event* event = nullptr);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
template <typename T>
void max(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
         const Queue& queue, Event* event = nullptr);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
template <typename T>
void min(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<unsigned int>& imin_buffer, const size_t imin_offset,
         const Queue& queue, Event* event = nullptr);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
template <typename T>
void gemv(const Layout layout, const Transpose a_transpose,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
template <typename T>
void gbmv(const Layout layout, const Transpose a_transpose,
          const size_t m, const size_t n, const size_t kl, const size_t ku,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
template <typename T>
void hemv(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
template <typename T>
void hbmv(const Layout layout, const Triangle triangle,
          const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
template <typename T>
void hpmv(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& ap_buffer, const size_t ap_offset,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
template <typename T>
void symv(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
template <typename T>
void sbmv(const Layout layout, const Triangle triangle,
          const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
template <typename T>
void spmv(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& ap_buffer, const size_t ap_offset,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event = nullptr);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
template <typename T>
void trmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event = nullptr);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
template <typename T>
void tbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n, const size_t k,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event = nullptr);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
template <typename T>
void tpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n,
          const Buffer<T>& ap_buffer, const size_t ap_offset,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event = nullptr);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
void trsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event = nullptr);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
template <typename T>
void tbsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n, const size_t k,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event = nullptr);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
template <typename T>
void tpsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n,
          const Buffer<T>& ap_buffer, const size_t ap_offset,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event = nullptr);

// General rank-1 matrix update: SGER/DGER/HGER
template <typename T>
void ger(const Layout layout,
         const size_t m, const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
         const Queue& queue, Event* event = nullptr);

// General rank-1 complex matrix update: CGERU/ZGERU
template <typename T>
void geru(const Layout layout,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event = nullptr);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
template <typename T>
void gerc(const Layout layout,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event = nullptr);

// Hermitian rank-1 matrix update: CHER/ZHER
template <typename T>
void her(const Layout layout, const Triangle triangle,
         const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
         const Queue& queue, Event* event = nullptr);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
template <typename T>
void hpr(const Layout layout, const Triangle triangle,
         const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& ap_buffer, const size_t ap_offset,
         const Queue& queue, Event* event = nullptr);

// Hermitian rank-2 matrix update: CHER2/ZHER2
template <typename T>
void her2(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event = nullptr);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
template <typename T>
void hpr2(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& ap_buffer, const size_t ap_offset,
          const Queue& queue, Event* event = nullptr);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
template <typename T>
void syr(const Layout layout, const Triangle triangle,
         const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
         const Queue& queue, Event* event = nullptr);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
template <typename T>
void spr(const Layout layout, const Triangle triangle,
         const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& ap_buffer, const size_t ap_offset,
         const Queue& queue, Event* event = nullptr);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
template <typename T>
void syr2(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event = nullptr);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
template <typename T>
void spr2(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& ap_buffer, const size_t ap_offset,
          const Queue& queue, Event* event = nullptr);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
template <typename T>
void gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
          const size_t m, const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event = nullptr,
          Buffer<T>* temp_buffer = nullptr);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
template <typename T>
void symm(const Layout layout, const Side side, const Triangle triangle,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event = nullptr);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
template <typename T>
void hemm(const Layout layout, const Side side, const Triangle triangle,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event = nullptr);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
template <typename T>
void syrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
          const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event = nullptr);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
template <typename T>
void herk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
          const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event = nullptr);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
template <typename T>
void syr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
           const size_t n, const size_t k,
           const T alpha,
           const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
           const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
           const T beta,
           Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
           const Queue& queue, Event* event = nullptr);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
template <typename T, typename U>
void her2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
           const size_t n, const size_t k,
           const T alpha,
           const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
           const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
           const U beta,
           Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
           const Queue& queue, Event* event = nullptr);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
template <typename T>
void trmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const Queue& queue, Event* event = nullptr);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
template <typename T>
void trsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const Queue& queue, Event* event = nullptr);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Element-wise vector product (Hadamard): SHAD/DHAD/CHAD/ZHAD/HHAD
template <typename T>
void had(const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         const T beta,
         Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc,
         const Queue& queue, Event* event = nullptr);

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
template <typename T>
void omatcopy(const Layout layout, const Transpose a_transpose,
              const size_t m, const size_t n,
              const T alpha,
              const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
              Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
              const Queue& queue, Event* event = nullptr);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
template <typename T>
void im2col(const KernelMode kernel_mode,
            const size_t channels, const size_t height, const size_t width,
            const size_t kernel_h, const size_t kernel_w,
            const size_t pad_h, const size_t pad_w,
            const size_t stride_h, const size_t stride_w,
            const size_t dilation_h, const size_t dilation_w,
            const Buffer<T>& im_buffer, const size_t im_offset,
            Buffer<T>& col_buffer, const size_t col_offset,
            const Queue& queue, Event* event = nullptr);

// Col2im function (non-BLAS function): SCOL2IM/DCOL2IM/CCOL2IM/ZCOL2IM/HCOL2IM
template <typename T>
void col2im(const KernelMode kernel_mode,
            const size_t channels, const size_t height, const size_t width,
            const size_t kernel_h, const size_t kernel_w,
            const size_t pad_h, const size_t pad_w,
            const size_t stride_h, const size_t stride_w,
            const size_t dilation_h, const size_t dilation_w,
            const Buffer<T>& col_buffer, const size_t col_offset,
            Buffer<T>& im_buffer, const size_t im_offset,
            const Queue& queue, Event* event = nullptr);

// Batched convolution as GEMM (non-BLAS function): SCONVGEMM/DCONVGEMM/HCONVGEMM
template <typename T>
void convgemm(const KernelMode kernel_mode,
              const size_t channels, const size_t height, const size_t width,
              const size_t kernel_h, const size_t kernel_w,
              const size_t pad_h, const size_t pad_w,
              const size_t stride_h, const size_t stride_w,
              const size_t dilation_h, const size_t dilation_w,
              const size_t num_kernels, const size_t batch_count,
              const Buffer<T>& im_buffer, const size_t im_offset,
              const Buffer<T>& kernel_buffer, const size_t kernel_offset,
              Buffer<T>& result_buffer, const size_t result_offset,
              const Queue& queue, Event* event = nullptr);

// Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED
template <typename T>
void axpyBatched(const size_t n,
                 const T *alphas,
                 const Buffer<T>& x_buffer, const size_t *x_offsets, const size_t x_inc,
                 Buffer<T>& y_buffer, const size_t *y_offsets, const size_t y_inc,
                 const size_t batch_count,
                 const Queue& queue, Event* event = nullptr);

// Batched version of GEMM: SGEMMBATCHED/DGEMMBATCHED/CGEMMBATCHED/ZGEMMBATCHED/HGEMMBATCHED
template <typename T>
void gemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                 const size_t m, const size_t n, const size_t k,
                 const T *alphas,
                 const Buffer<T>& a_buffer, const size_t *a_offsets, const size_t a_ld,
                 const Buffer<T>& b_buffer, const size_t *b_offsets, const size_t b_ld,
                 const T *betas,
                 Buffer<T>& c_buffer, const size_t *c_offsets, const size_t c_ld,
                 const size_t batch_count,
                 const Queue& queue, Event* event = nullptr);

// StridedBatched version of GEMM: SGEMMSTRIDEDBATCHED/DGEMMSTRIDEDBATCHED/CGEMMSTRIDEDBATCHED/ZGEMMSTRIDEDBATCHED/HGEMMSTRIDEDBATCHED
template <typename T>
void gemmStridedBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const T alpha,
                        const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                        const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                        const T beta,
                        Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                        const size_t batch_count,
                        const Queue& queue, Event* event = nullptr);

// Vector clamp (non-BLAS function): SCLAMP/DCLAMP/HCLAMP
template <typename T>
void clamp(const size_t n, const T minval, const T maxval,
           Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
           const Queue& queue, Event* event = nullptr);

// =================================================================================================

// Retrieves the required size of the temporary buffer for the GEMM kernel (optional)
template <typename T>
size_t gemmTempBufferSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                          const size_t m, const size_t n, const size_t k,
                          const size_t a_offset, const size_t a_ld,
                          const size_t b_offset, const size_t b_ld,
                          const size_t c_offset, const size_t c_ld,
                          const Queue& queue);

// =================================================================================================

// CLBlast stores binaries of compiled kernels into a cache in case the same kernel is used later on
// for the same device. This cache can be cleared to free up system memory or in case of debugging.
void PUBLIC_API ClearCache();

// The cache can also be pre-initialized for a specific device with all possible CLBLast kernels.
// Further CLBlast routine calls will then run at maximum speed.
void PUBLIC_API FillCache(const Device& device);

// =================================================================================================

// Retrieves current tuning parameters for a specific device-precision-kernel combination
void PUBLIC_API RetrieveParameters(const Device& device, const std::string& kernel_name,
                                   const Precision precision,
                                   std::unordered_map<std::string,size_t>& parameters);

// Overrides tuning parameters for a specific device-precision-kernel combination. The next time
// the target routine is called it will re-compile and use the new parameters from then on.
void PUBLIC_API OverrideParameters(const Device& device, const std::string& kernel_name,
                                   const Precision precision,
                                   const std::unordered_map<std::string,size_t>& parameters);

}} // namespace gpgpu::blas

#endif //GPGPU_BLAS_H_
