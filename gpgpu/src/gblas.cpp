
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements all the BLAS API calls. In all cases, it does not much more than creating
// a new object of the appropriate type, and calling the main routine on that object. It forwards
// all status codes to the caller.
//
// =================================================================================================

#include <string>

#include "gblas.h"
#include "routines/routines.hpp"
#include "cublas.hpp"

namespace gpgpu { namespace blas {

// =================================================================================================
// cuBlas integration
// =================================================================================================

#if HAS_CUDA
using namespace gpgpu::cu;

template <typename T>
constexpr cudaDataType CudaDataType = static_cast<cudaDataType>(-1);

template <> constexpr cudaDataType CudaDataType<float>   = cudaDataType::CUDA_R_32F;
template <> constexpr cudaDataType CudaDataType<double>  = cudaDataType::CUDA_R_64F;
template <> constexpr cudaDataType CudaDataType<float2>  = cudaDataType::CUDA_C_32F;
template <> constexpr cudaDataType CudaDataType<double2> = cudaDataType::CUDA_C_64F;
template <> constexpr cudaDataType CudaDataType<half>    = cudaDataType::CUDA_R_16F;

template <typename T>
constexpr bool RequireCublas =
    !(std::is_integral<T>::value || std::is_same<T, half>::value);

static cublasOperation_t CudaOp(Transpose trans) {
    switch (trans) {
    case Transpose::NoTrans:
        return cublasOperation_t::CUBLAS_OP_N;
    case Transpose::Trans:
        return cublasOperation_t::CUBLAS_OP_T;
    case Transpose::ConjTrans:
        return cublasOperation_t::CUBLAS_OP_C;
    }
}

template <typename T, typename CL, typename CU>
inline std::enable_if_t<RequireCublas<T>>
dispatch(const Queue& queue, CL&& cl, CU&& cu) {
    if (IsOpenCL(queue.context().device())) {
        cl();
    } else {
        cu(static_cast<const cuQueue&>(queue.raw()));
    }
}

template <typename T, typename CL, typename CU>
inline std::enable_if_t<!RequireCublas<T>>
dispatch(const Queue&, CL&& cl, CU&&) {
    cl();
}

#define OPENCL(...) [&](){__VA_ARGS__}

#define CUBLAS(...) [&](const auto& q) { \
    auto h = q.getCublasHandle(); \
    __VA_ARGS__ \
}

#define CUSOLVER(...) [&](const auto& q) { \
    auto h = q.getCusolverHandle(); \
    __VA_ARGS__ \
}

#else

template <typename T, typename CL, typename CU>
void dispatch(const Queue&, CL&& cl, CU&&) {
    cl();
}

#define OPENCL(...)   [&](){__VA_ARGS__}
#define CUBLAS(...)   [&](){}
#define CUSOLVER(...) [&](){}

#endif // !HAS_CUDA

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

//---------------------------------------------------------------------------
// Generate givens plane rotation: SROTG/DROTG

template <typename T>
void rotg(Buffer<T>&, const size_t,
          Buffer<T>&, const size_t,
          Buffer<T>&, const size_t,
          Buffer<T>&, const size_t,
          const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API rotg<float> (Buffer<float>&, const size_t,
                                      Buffer<float>&, const size_t,
                                      Buffer<float>&, const size_t,
                                      Buffer<float>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API rotg<double>(Buffer<double>&, const size_t,
                                      Buffer<double>&, const size_t,
                                      Buffer<double>&, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Generate modified givens plane rotation: SROTMG/DROTMG

template <typename T>
void rotmg(Buffer<T>&, const size_t,
           Buffer<T>&, const size_t,
           Buffer<T>&, const size_t,
           const Buffer<T>&, const size_t,
           Buffer<T>&, const size_t,
           const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API rotmg<float> (Buffer<float>&, const size_t,
                                       Buffer<float>&, const size_t,
                                       Buffer<float>&, const size_t,
                                       const Buffer<float>&, const size_t,
                                       Buffer<float>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API rotmg<double>(Buffer<double>&, const size_t,
                                       Buffer<double>&, const size_t,
                                       Buffer<double>&, const size_t,
                                       const Buffer<double>&, const size_t,
                                       Buffer<double>&, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Apply givens plane rotation: SROT/DROT

template <typename T>
void rot(const size_t,
         Buffer<T>&, const size_t, const size_t,
         Buffer<T>&, const size_t, const size_t,
         const T, const T,
         const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API rot<float> (const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const float, const float,
                                     const Queue&, Event*);
template void PUBLIC_API rot<double>(const size_t,
                                     Buffer<double>&, const size_t, const size_t,
                                     Buffer<double>&, const size_t, const size_t,
                                     const double, const double,
                                     const Queue&, Event*);

//---------------------------------------------------------------------------
// Apply modified givens plane rotation: SROTM/DROTM

template <typename T>
void rotm(const size_t,
          Buffer<T>&, const size_t, const size_t,
          Buffer<T>&, const size_t, const size_t,
          Buffer<T>&, const size_t,
          const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API rotm<float> (const size_t,
                                      Buffer<float>&, const size_t, const size_t,
                                      Buffer<float>&, const size_t, const size_t,
                                      Buffer<float>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API rotm<double>(const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP

template <typename T>
void swap(const size_t n,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xswap<T>(queue, event);
            routine.DoSwap(n,
                           x_buffer, x_offset, x_inc,
                           y_buffer, y_offset, y_inc);
        ),
        CUBLAS(
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            auto y = cuBuffer::unwrap(y_buffer) + y_offset;
            cublasSwapEx(h, n, x, x_inc, y, y_inc);
        ));
}

template void PUBLIC_API swap<float>  (const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API swap<double> (const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API swap<float2> (const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API swap<double2>(const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API swap<half>   (const size_t,
                                       Buffer<half>&, const size_t, const size_t,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API swap<int32_t>(const size_t,
                                       Buffer<int32_t>&, const size_t, const size_t,
                                       Buffer<int32_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API swap<int64_t>(const size_t,
                                       Buffer<int64_t>&, const size_t, const size_t,
                                       Buffer<int64_t>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL

template <typename T>
void scal(const size_t n, const T alpha,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xscal<T>(queue, event);
            routine.DoScal(n, alpha, x_buffer, x_offset, x_inc);
        ),
        CUBLAS(
            auto t = CudaDataType<T>;
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            cublasScalEx(h, n, &alpha, t, x, t, x_inc, t);
        ));
}

template void PUBLIC_API scal<float>  (const size_t, const float,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API scal<double> (const size_t, const double,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API scal<float2> (const size_t, const float2,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API scal<double2>(const size_t, const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API scal<half>   (const size_t, const half,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API scal<int32_t>(const size_t, const int32_t,
                                       Buffer<int32_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API scal<int64_t>(const size_t, const int64_t,
                                       Buffer<int64_t>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY

template <typename T>
void copy(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xcopy<T>(queue, event);
            routine.DoCopy(n,
                           x_buffer, x_offset, x_inc,
                           y_buffer, y_offset, y_inc);
        ),
        CUBLAS(
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            auto y = cuBuffer::unwrap(y_buffer) + y_offset;
            cublasCopyEx(h, n, x, x_inc, y, y_inc);
        ));
}

template void PUBLIC_API copy<float>  (const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double> (const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float2> (const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<half>   (const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY

template <typename T>
void axpy(const size_t n, const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xaxpy<T>(queue, event);
            routine.DoAxpy(n, alpha,
                           x_buffer, x_offset, x_inc,
                           y_buffer, y_offset, y_inc);
        ),
        CUBLAS(
            auto t = CudaDataType<T>;
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            auto y = cuBuffer::unwrap(y_buffer) + y_offset;
            cublasAxpyEx(h, n, &alpha, t, x, t, x_inc, y, t, y_inc, t);
        ));
}

template void PUBLIC_API axpy<float>  (const size_t, const float,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API axpy<double> (const size_t, const double,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API axpy<float2> (const size_t, const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API axpy<double2>(const size_t, const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API axpy<half>   (const size_t, const half,
                                       const Buffer<half>&, const size_t, const size_t,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API axpy<int32_t>(const size_t, const int32_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       Buffer<int32_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API axpy<int64_t>(const size_t, const int64_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       Buffer<int64_t>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Dot product of two vectors: SDOT/DDOT/HDOT

template <typename T>
void dot(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& r_buffer, const size_t r_offset,
         const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xdot<T>(queue, event);
            routine.DoDot(n,
                          x_buffer, x_offset, x_inc,
                          y_buffer, y_offset, y_inc,
                          r_buffer, r_offset);
        ),
        CUBLAS(
            auto t = CudaDataType<T>;
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            auto y = cuBuffer::unwrap(y_buffer) + y_offset;
            auto r = cuBuffer::unwrap(r_buffer) + r_offset;

            cublasPointerMode_t mode;
            cublasGetPointerMode(h, &mode);
            cublasSetPointerMode(h, cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE);
            cublasDotEx(h, n, x, t, x_inc, y, t, y_inc, r, t, t);
            cublasSetPointerMode(h, mode);
        ));
}

template void PUBLIC_API dot<float>  (const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      Buffer<float>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API dot<double> (const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API dot<half>   (const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      Buffer<half>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API dot<int32_t>(const size_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      Buffer<int32_t>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API dot<int64_t>(const size_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      Buffer<int64_t>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API dot<float2>  (const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API dot<double2> (const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC

template <typename T>
void dotc(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& r_buffer, const size_t r_offset,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xdot<T>(queue, event);
            routine.DoDot(n,
                          x_buffer, x_offset, x_inc,
                          y_buffer, y_offset, y_inc,
                          r_buffer, r_offset,
                          true);
        ),
        CUBLAS(
            auto t = CudaDataType<T>;
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            auto y = cuBuffer::unwrap(y_buffer) + y_offset;
            auto r = cuBuffer::unwrap(r_buffer) + r_offset;
            cublasDotcEx(h, n, x, t, x_inc, y, t, y_inc, r, t, t);
        ));
}

template void PUBLIC_API dotc<float2> (const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API dotc<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2

template <typename T>
void nrm2(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& r_buffer, const size_t r_offset,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xnrm2<T>(queue, event);
            routine.DoNrm2(n,
                           x_buffer, x_offset, x_inc,
                           r_buffer, r_offset);
        ),
        CUBLAS(
            auto t = CudaDataType<T>;
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            auto r = cuBuffer::unwrap(r_buffer) + r_offset;

            cublasPointerMode_t mode;
            cublasGetPointerMode(h, &mode);
            cublasSetPointerMode(h, cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE);
            cublasNrm2Ex(h, n, x, t, x_inc, r, t, t);
            cublasSetPointerMode(h, mode);
        ));
}

template void PUBLIC_API nrm2<float>  (const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API nrm2<double> (const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API nrm2<float2> (const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API nrm2<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API nrm2<half>   (const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       Buffer<half>&, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM

template <typename T>
void asum(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& r_buffer, const size_t r_offset,
          const Queue& queue, Event* event) {
    auto routine = Xasum<T>(queue, event);
    routine.DoAsum(n,
                   x_buffer, x_offset, x_inc,
                   r_buffer, r_offset);
}

template void PUBLIC_API asum<float>  (const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API asum<double> (const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API asum<float2> (const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API asum<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API asum<half>   (const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       Buffer<half>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API asum<int32_t>(const size_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       Buffer<int32_t>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API asum<int64_t>(const size_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       Buffer<int64_t>&, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM

template <typename T>
void sum(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& r_buffer, const size_t r_offset,
         const Queue& queue, Event* event) {
    auto routine = Xsum<T>(queue, event);
    routine.DoSum(n,
                  x_buffer, x_offset, x_inc,
                  r_buffer, r_offset);
}

template void PUBLIC_API sum<float>  (const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      Buffer<float>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API sum<double> (const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API sum<float2> (const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API sum<double2>(const size_t,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      Buffer<double2>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API sum<half>   (const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      Buffer<half>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API sum<int32_t>(const size_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      Buffer<int32_t>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API sum<int64_t>(const size_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      Buffer<int64_t>&, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX

template <typename T>
void amax(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<unsigned int>& r_buffer, const size_t r_offset,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xamax<T>(queue, event);
            routine.DoAmax(n,
                           x_buffer, x_offset, x_inc,
                           r_buffer, r_offset);
        ),
        CUBLAS(
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            auto r = cuBuffer::unwrap(r_buffer) + r_offset;

            cublasPointerMode_t mode;
            cublasGetPointerMode(h, &mode);
            cublasSetPointerMode(h, cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE);
            cublasAmaxEx(h, n, x, x_inc, reinterpret_cast<int*>(r));
            cublasSetPointerMode(h, mode);
        ));
}

template void PUBLIC_API amax<float>  (const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amax<double> (const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amax<float2> (const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amax<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amax<half>   (const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amax<int32_t>(const size_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amax<int64_t>(const size_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN

template <typename T>
void amin(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<unsigned int>& r_buffer, const size_t r_offset,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xamin<T>(queue, event);
            routine.DoAmin(n,
                           x_buffer, x_offset, x_inc,
                           r_buffer, r_offset);
        ),
        CUBLAS(
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            auto r = cuBuffer::unwrap(r_buffer) + r_offset;

            cublasPointerMode_t mode;
            cublasGetPointerMode(h, &mode);
            cublasSetPointerMode(h, cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE);
            cublasAminEx(h, n, x, x_inc, reinterpret_cast<int*>(r));
            cublasSetPointerMode(h, mode);
        ));
}

template void PUBLIC_API amin<float>  (const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amin<double> (const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amin<float2> (const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amin<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amin<half>   (const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amin<int32_t>(const size_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amin<int64_t>(const size_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX

template <typename T>
void max(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<unsigned int>& r_buffer, const size_t r_offset,
         const Queue& queue, Event* event) {
    auto routine = Xmax<T>(queue, event);
    routine.DoMax(n,
                  x_buffer, x_offset, x_inc,
                  r_buffer, r_offset);
}

template void PUBLIC_API max<float>  (const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API max<double> (const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API max<float2> (const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API max<double2>(const size_t,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API max<half>   (const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API max<int32_t>(const size_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API max<int64_t>(const size_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN

template <typename T>
void min(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<unsigned int>& r_buffer, const size_t r_offset,
         const Queue& queue, Event* event) {
    auto routine = Xmin<T>(queue, event);
    routine.DoMin(n,
                  x_buffer, x_offset, x_inc,
                  r_buffer, r_offset);
}

template void PUBLIC_API min<float>  (const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API min<double> (const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API min<float2> (const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API min<double2>(const size_t,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API min<half>   (const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API min<int32_t>(const size_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API min<int64_t>(const size_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

//---------------------------------------------------------------------------
// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV

template <typename T>
void gemv(const Layout layout, const Transpose a_transpose,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event)
{
    if (layout == Layout::RowMajor && a_transpose == Transpose::ConjTrans) {
        auto routine = Xgemv<T>(queue, event);
        routine.DoGemv(layout, a_transpose,
                       m, n,
                       alpha,
                       a_buffer, a_offset, a_ld,
                       x_buffer, x_offset, x_inc,
                       beta,
                       y_buffer, y_offset, y_inc);
        return;
    }

    dispatch<T>(queue,
        OPENCL(
            auto routine = Xgemv<T>(queue, event);
            routine.DoGemv(layout, a_transpose,
                           m, n,
                           alpha,
                           a_buffer, a_offset, a_ld,
                           x_buffer, x_offset, x_inc,
                           beta,
                           y_buffer, y_offset, y_inc);
        ),
        CUBLAS(
            auto a = cuBuffer::unwrap(a_buffer) + a_offset;
            auto x = cuBuffer::unwrap(x_buffer) + x_offset;
            auto y = cuBuffer::unwrap(y_buffer) + y_offset;

            if (layout == Layout::RowMajor) {
                auto transA = a_transpose == Transpose::NoTrans
                    ? cublasOperation_t::CUBLAS_OP_T
                    : cublasOperation_t::CUBLAS_OP_N;
                cublasGemvEx(h, transA, n, m, alpha, a, a_ld, x, x_inc, beta, y, y_inc);
            } else {
                auto transA = CudaOp(a_transpose);
                cublasGemvEx(h, transA, m, n, alpha, a, a_ld, x, x_inc, beta, y, y_inc);
            }
        ));
}

template void PUBLIC_API gemv<float>  (const Layout, const Transpose,
                                       const size_t, const size_t,
                                       const float,
                                       const Buffer<float>&, const size_t, const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       const float,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gemv<double> (const Layout, const Transpose,
                                       const size_t, const size_t,
                                       const double,
                                       const Buffer<double>&, const size_t, const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       const double,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gemv<float2> (const Layout, const Transpose,
                                       const size_t, const size_t,
                                       const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const float2,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gemv<double2>(const Layout, const Transpose,
                                       const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gemv<half>   (const Layout, const Transpose,
                                       const size_t, const size_t,
                                       const half,
                                       const Buffer<half>&, const size_t, const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       const half,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gemv<int32_t>(const Layout, const Transpose,
                                       const size_t, const size_t,
                                       const int32_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       const int32_t,
                                       Buffer<int32_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gemv<int64_t>(const Layout, const Transpose,
                                       const size_t, const size_t,
                                       const int64_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       const int64_t,
                                       Buffer<int64_t>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV

template <typename T>
void gbmv(const Layout layout, const Transpose a_transpose,
          const size_t m, const size_t n, const size_t kl, const size_t ku,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xgbmv<T>(queue, event);
    routine.DoGbmv(layout, a_transpose,
                   m, n, kl, ku,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API gbmv<float>  (const Layout, const Transpose,
                                       const size_t, const size_t, const size_t, const size_t,
                                       const float,
                                       const Buffer<float>&, const size_t, const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       const float,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gbmv<double> (const Layout, const Transpose,
                                       const size_t, const size_t, const size_t, const size_t,
                                       const double,
                                       const Buffer<double>&, const size_t, const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       const double,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gbmv<float2> (const Layout, const Transpose,
                                       const size_t, const size_t, const size_t, const size_t,
                                       const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const float2,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gbmv<double2>(const Layout, const Transpose,
                                       const size_t, const size_t, const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gbmv<half>   (const Layout, const Transpose,
                                       const size_t, const size_t, const size_t, const size_t,
                                       const half,
                                       const Buffer<half>&, const size_t, const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       const half,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Hermitian matrix-vector multiplication: CHEMV/ZHEMV

template <typename T>
void hemv(const Layout layout, const Triangle triangle, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xhemv<T>(queue, event);
    routine.DoHemv(layout, triangle, n,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API hemv<float2> (const Layout, const Triangle, const size_t,
                                       const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const float2,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API hemv<double2>(const Layout, const Triangle, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV

template <typename T>
void hbmv(const Layout layout, const Triangle triangle,
          const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xhbmv<T>(queue, event);
    routine.DoHbmv(layout, triangle,
                   n, k,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API hbmv<float2> (const Layout, const Triangle,
                                       const size_t, const size_t,
                                       const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const float2,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API hbmv<double2>(const Layout, const Triangle,
                                       const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV

template <typename T>
void hpmv(const Layout layout, const Triangle triangle, const size_t n,
          const T alpha,
          const Buffer<T>& ap_buffer, const size_t ap_offset,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xhpmv<T>(queue, event);
    routine.DoHpmv(layout, triangle, n,
                   alpha,
                   ap_buffer, ap_offset,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API hpmv<float2> (const Layout, const Triangle, const size_t,
                                       const float2,
                                       const Buffer<float2>&, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const float2,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API hpmv<double2>(const Layout, const Triangle, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV

template <typename T>
void symv(const Layout layout, const Triangle triangle, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xsymv<T>(queue, event);
            routine.DoSymv(layout, triangle, n,
                           alpha,
                           a_buffer, a_offset, a_ld,
                           x_buffer, x_offset, x_inc,
                           beta,
                           y_buffer, y_offset, y_inc);
        ),
        CUBLAS(
            if (layout == Layout::RowMajor) {
                auto uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
                cublasSymv(h, uplo, n, alpha,
                           cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                           cuBuffer::unwrap(x_buffer) + x_offset, x_inc,
                           beta,
                           cuBuffer::unwrap(y_buffer) + y_offset, y_inc);
            } else {
                auto uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
                cublasSymv(h, uplo, n, alpha,
                           cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                           cuBuffer::unwrap(x_buffer) + x_offset, x_inc,
                           beta,
                           cuBuffer::unwrap(y_buffer) + y_offset, y_inc);
            }
        ));
}

template void PUBLIC_API symv<float> (const Layout, const Triangle, const size_t,
                                      const float,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const float,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API symv<double>(const Layout, const Triangle, const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API symv<half>  (const Layout, const Triangle, const size_t,
                                      const half,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const half,
                                      Buffer<half>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API symv<int32_t>(const Layout, const Triangle, const size_t,
                                      const int32_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      const int32_t,
                                      Buffer<int32_t>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API symv<int64_t>(const Layout, const Triangle, const size_t,
                                      const int64_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      const int64_t,
                                      Buffer<int64_t>&, const size_t, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV

template <typename T>
void sbmv(const Layout layout, const Triangle triangle,
          const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xsbmv<T>(queue, event);
    routine.DoSbmv(layout, triangle, n, k,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API sbmv<float> (const Layout, const Triangle,
                                      const size_t, const size_t,
                                      const float,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const float,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API sbmv<double>(const Layout, const Triangle,
                                      const size_t, const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API sbmv<half>  (const Layout, const Triangle,
                                      const size_t, const size_t,
                                      const half,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const half,
                                      Buffer<half>&, const size_t, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV

template <typename T>
void spmv(const Layout layout, const Triangle triangle, const size_t n,
          const T alpha,
          const Buffer<T>& ap_buffer, const size_t ap_offset,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xspmv<T>(queue, event);
    routine.DoSpmv(layout, triangle, n,
                   alpha,
                   ap_buffer, ap_offset,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API spmv<float> (const Layout, const Triangle, const size_t,
                                      const float,
                                      const Buffer<float>&, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const float,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API spmv<double>(const Layout, const Triangle, const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API spmv<half>  (const Layout, const Triangle, const size_t,
                                      const half,
                                      const Buffer<half>&, const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const half,
                                      Buffer<half>&, const size_t, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV

template <typename T>
void trmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
                Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                const Queue& queue, Event* event)
{
    if (layout == Layout::RowMajor && a_transpose == Transpose::ConjTrans) {
        auto routine = Xtrmv<T>(queue, event);
        routine.DoTrmv(layout, triangle, a_transpose, diagonal, n,
                       a_buffer, a_offset, a_ld,
                       x_buffer, x_offset, x_inc);
        return;
    }

    dispatch<T>(queue,
        OPENCL(
            auto routine = Xtrmv<T>(queue, event);
            routine.DoTrmv(layout, triangle, a_transpose, diagonal, n,
                           a_buffer, a_offset, a_ld,
                           x_buffer, x_offset, x_inc);
        ),
        CUBLAS(
            if (layout == Layout::RowMajor) {
                auto uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
                auto trans = a_transpose == Transpose::NoTrans
                    ? cublasOperation_t::CUBLAS_OP_T
                    : cublasOperation_t::CUBLAS_OP_N;
                auto diag = diagonal == Diagonal::NonUnit
                    ? cublasDiagType_t::CUBLAS_DIAG_NON_UNIT
                    : cublasDiagType_t::CUBLAS_DIAG_UNIT;
                cublasTrmv(h, uplo, trans, diag, n,
                           cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                           cuBuffer::unwrap(x_buffer) + x_offset, x_inc);
            } else {
                auto uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
                auto diag = diagonal == Diagonal::NonUnit
                    ? cublasDiagType_t::CUBLAS_DIAG_NON_UNIT
                    : cublasDiagType_t::CUBLAS_DIAG_UNIT;
                cublasTrmv(h, uplo, CudaOp(a_transpose), diag, n,
                           cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                           cuBuffer::unwrap(x_buffer) + x_offset, x_inc);
            }
        ));
}

template void PUBLIC_API trmv<float>  (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmv<double> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmv<float2> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmv<half>   (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmv<int32_t>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       Buffer<int32_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmv<int64_t>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       Buffer<int64_t>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV

template <typename T>
void tbmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n, const size_t k,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event) {
    auto routine = Xtbmv<T>(queue, event);
    routine.DoTbmv(layout, triangle, a_transpose, diagonal,
                   n, k,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc);
}

template void PUBLIC_API tbmv<float>  (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tbmv<double> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tbmv<float2> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tbmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tbmv<half>   (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV

template <typename T>
void tpmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n,
          const Buffer<T>& ap_buffer, const size_t ap_offset,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event) {
    auto routine = Xtpmv<T>(queue, event);
    routine.DoTpmv(layout, triangle, a_transpose, diagonal,
                   n,
                   ap_buffer, ap_offset,
                   x_buffer, x_offset, x_inc);
}

template void PUBLIC_API tpmv<float>  (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<float>&, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tpmv<double> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double>&, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tpmv<float2> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<float2>&, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tpmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double2>&, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tpmv<half>   (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<half>&, const size_t,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV

template <typename T>
void trsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event)
{
    if (layout == Layout::RowMajor && a_transpose == Transpose::ConjTrans) {
        auto routine = Xtrsv<T>(queue, event);
        routine.DoTrsv(layout, triangle, a_transpose, diagonal, n,
                       a_buffer, a_offset, a_ld,
                       x_buffer, x_offset, x_inc);
        return;
    }

    dispatch<T>(queue,
        OPENCL(
            auto routine = Xtrsv<T>(queue, event);
            routine.DoTrsv(layout, triangle, a_transpose, diagonal, n,
                           a_buffer, a_offset, a_ld,
                           x_buffer, x_offset, x_inc);
        ),
        CUBLAS(
            if (layout == Layout::RowMajor) {
                auto uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
                auto trans = a_transpose == Transpose::NoTrans
                    ? cublasOperation_t::CUBLAS_OP_T
                    : cublasOperation_t::CUBLAS_OP_N;
                auto diag = diagonal == Diagonal::NonUnit
                    ? cublasDiagType_t::CUBLAS_DIAG_NON_UNIT
                    : cublasDiagType_t::CUBLAS_DIAG_UNIT;
                cublasTrsv(h, uplo, trans, diag, n,
                           cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                           cuBuffer::unwrap(x_buffer) + x_offset, x_inc);
            } else {
                auto uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
                auto diag = diagonal == Diagonal::NonUnit
                    ? cublasDiagType_t::CUBLAS_DIAG_NON_UNIT
                    : cublasDiagType_t::CUBLAS_DIAG_UNIT;
                cublasTrsv(h, uplo, CudaOp(a_transpose), diag, n,
                           cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                           cuBuffer::unwrap(x_buffer) + x_offset, x_inc);
            }
        ));
}

template void PUBLIC_API trsv<float>  (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trsv<double> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trsv<float2> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV

template <typename T>
void tbsv(const Layout, const Triangle, const Transpose, const Diagonal,
          const size_t, const size_t,
          const Buffer<T>&, const size_t, const size_t,
          Buffer<T>&, const size_t, const size_t,
          const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API tbsv<float>  (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tbsv<double> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tbsv<float2> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tbsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV

template <typename T>
void tpsv(const Layout, const Triangle, const Transpose, const Diagonal,
          const size_t,
          const Buffer<T>&, const size_t,
          Buffer<T>&, const size_t, const size_t,
          const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API tpsv<float>  (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<float>&, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tpsv<double> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double>&, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tpsv<float2> (const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<float2>&, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tpsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double2>&, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// General rank-1 matrix update: SGER/DGER/HGER

template <typename T>
void ger(const Layout layout, const size_t m, const size_t n, const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
         const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xger<T>(queue, event);
            routine.DoGer(layout, m, n, alpha,
                          x_buffer, x_offset, x_inc,
                          y_buffer, y_offset, y_inc,
                          a_buffer, a_offset, a_ld);
        ),
        CUBLAS(
            cublasGer(h, n, m, alpha,
                      cuBuffer::unwrap(y_buffer) + y_offset, y_inc,
                      cuBuffer::unwrap(x_buffer) + x_offset, x_inc,
                      cuBuffer::unwrap(a_buffer) + a_offset, a_ld);
        ));
}

template <typename T>
void gerc(const Layout layout, const size_t m, const size_t n, const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xgerc<T>(queue, event);
            routine.DoGerc(layout, m, n, alpha,
                           x_buffer, x_offset, x_inc,
                           y_buffer, y_offset, y_inc,
                           a_buffer, a_offset, a_ld);
        ),
        CUBLAS(
            cublasGerc(h, n, m, alpha,
                       cuBuffer::unwrap(y_buffer) + y_offset, y_inc,
                       cuBuffer::unwrap(x_buffer) + x_offset, x_inc,
                       cuBuffer::unwrap(a_buffer) + a_offset, a_ld);
        ));
}

template void PUBLIC_API ger<float>  (const Layout, const size_t, const size_t, const float,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API ger<double> (const Layout, const size_t, const size_t, const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API ger<half>   (const Layout, const size_t, const size_t, const half,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      Buffer<half>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API ger<float2> (const Layout, const size_t, const size_t, const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API ger<double2>(const Layout, const size_t, const size_t, const double2,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      Buffer<double2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API ger<int32_t>(const Layout, const size_t, const size_t, const int32_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      Buffer<int32_t>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API ger<int64_t>(const Layout, const size_t, const size_t, const int64_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      Buffer<int64_t>&, const size_t, const size_t,
                                      const Queue&, Event*);

template void PUBLIC_API gerc<float2> (const Layout, const size_t, const size_t, const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API gerc<double2>(const Layout, const size_t, const size_t, const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Hermitian rank-1 matrix update: CHER/ZHER

template <typename T>
void her(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
         const Buffer<std::complex<T>>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<std::complex<T>>& a_buffer, const size_t a_offset, const size_t a_ld,
         const Queue& queue, Event* event) {
    auto routine = Xher<std::complex<T>,T>(queue, event);
    routine.DoHer(layout, triangle, n, alpha,
                  x_buffer, x_offset, x_inc,
                  a_buffer, a_offset, a_ld);
}

template void PUBLIC_API her<float> (const Layout, const Triangle, const size_t, const float,
                                     const Buffer<std::complex<float>>&, const size_t, const size_t,
                                     Buffer<std::complex<float>>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API her<double>(const Layout, const Triangle, const size_t, const double,
                                     const Buffer<std::complex<double>>&, const size_t, const size_t,
                                     Buffer<std::complex<double>>&, const size_t, const size_t,
                                     const Queue&, Event*);

//---------------------------------------------------------------------------
// Hermitian packed rank-1 matrix update: CHPR/ZHPR

template <typename T>
void hpr(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
         const Buffer<std::complex<T>>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<std::complex<T>>& ap_buffer, const size_t ap_offset,
         const Queue& queue, Event* event) {
    auto routine = Xhpr<std::complex<T>,T>(queue, event);
    routine.DoHpr(layout, triangle, n, alpha,
                  x_buffer, x_offset, x_inc,
                  ap_buffer, ap_offset);
}

template void PUBLIC_API hpr<float> (const Layout, const Triangle, const size_t, const float,
                                     const Buffer<std::complex<float>>&, const size_t, const size_t,
                                     Buffer<std::complex<float>>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API hpr<double>(const Layout, const Triangle, const size_t, const double,
                                     const Buffer<std::complex<double>>&, const size_t, const size_t,
                                     Buffer<std::complex<double>>&, const size_t,
                                     const Queue&, Event*);

//---------------------------------------------------------------------------
// Hermitian rank-2 matrix update: CHER2/ZHER2

template <typename T>
void her2(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event) {
    auto routine = Xher2<T>(queue, event);
    routine.DoHer2(layout, triangle, n, alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   a_buffer, a_offset, a_ld);
}

template void PUBLIC_API her2<float2> (const Layout, const Triangle, const size_t, const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API her2<double2>(const Layout, const Triangle, const size_t, const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2

template <typename T>
void hpr2(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& ap_buffer, const size_t ap_offset,
          const Queue& queue, Event* event) {
    auto routine = Xhpr2<T>(queue, event);
    routine.DoHpr2(layout, triangle, n, alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   ap_buffer, ap_offset);
}

template void PUBLIC_API hpr2<float2> (const Layout, const Triangle, const size_t, const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API hpr2<double2>(const Layout, const Triangle, const size_t, const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR

template <typename T>
void syr(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
         const Queue& queue, Event* event) {
    auto routine = Xsyr<T>(queue, event);
    routine.DoSyr(layout, triangle, n, alpha,
                  x_buffer, x_offset, x_inc,
                  a_buffer, a_offset, a_ld);
}

template void PUBLIC_API syr<float> (const Layout, const Triangle, const size_t, const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API syr<double>(const Layout, const Triangle, const size_t, const double,
                                     const Buffer<double>&, const size_t, const size_t,
                                     Buffer<double>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API syr<half>  (const Layout, const Triangle, const size_t, const half,
                                     const Buffer<half>&, const size_t, const size_t,
                                     Buffer<half>&, const size_t, const size_t,
                                     const Queue&, Event*);

//---------------------------------------------------------------------------
// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR

template <typename T>
void spr(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& ap_buffer, const size_t ap_offset,
         const Queue& queue, Event* event) {
    auto routine = Xspr<T>(queue, event);
    routine.DoSpr(layout, triangle, n, alpha,
                  x_buffer, x_offset, x_inc,
                  ap_buffer, ap_offset);
}

template void PUBLIC_API spr<float> (const Layout, const Triangle, const size_t, const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API spr<double>(const Layout, const Triangle, const size_t, const double,
                                     const Buffer<double>&, const size_t, const size_t,
                                     Buffer<double>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API spr<half>  (const Layout, const Triangle, const size_t, const half,
                                     const Buffer<half>&, const size_t, const size_t,
                                     Buffer<half>&, const size_t,
                                     const Queue&, Event*);

//---------------------------------------------------------------------------
// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2

template <typename T>
void syr2(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event) {
    auto routine = Xsyr2<T>(queue, event);
    routine.DoSyr2(layout, triangle, n, alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   a_buffer, a_offset, a_ld);
}

template void PUBLIC_API syr2<float> (const Layout, const Triangle, const size_t, const float,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API syr2<double>(const Layout, const Triangle, const size_t, const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API syr2<half>  (const Layout, const Triangle, const size_t, const half,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      Buffer<half>&, const size_t, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2

template <typename T>
void spr2(const Layout layout, const Triangle triangle, const size_t n, const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& ap_buffer, const size_t ap_offset,
          const Queue& queue, Event* event) {
    auto routine = Xspr2<T>(queue, event);
    routine.DoSpr2(layout, triangle, n, alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   ap_buffer, ap_offset);
}

template void PUBLIC_API spr2<float> (const Layout, const Triangle, const size_t, const float,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      Buffer<float>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API spr2<double>(const Layout, const Triangle, const size_t, const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API spr2<half>  (const Layout, const Triangle, const size_t, const half,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      Buffer<half>&, const size_t,
                                      const Queue&, Event*);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

//---------------------------------------------------------------------------
// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM

template <typename T>
void gemm(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
          const size_t m, const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xgemm<T>(queue, event);
            routine.DoGemm(layout, a_transpose, b_transpose,
                           m, n, k,
                           alpha,
                           a_buffer, a_offset, a_ld,
                           b_buffer, b_offset, b_ld,
                           beta,
                           c_buffer, c_offset, c_ld);
        ),
        CUBLAS(
            auto t = CudaDataType<T>;
            auto a = cuBuffer::unwrap(a_buffer) + a_offset;
            auto b = cuBuffer::unwrap(b_buffer) + b_offset;
            auto c = cuBuffer::unwrap(c_buffer) + c_offset;

            if (layout == Layout::RowMajor) {
                cublasGemmEx(h, CudaOp(b_transpose), CudaOp(a_transpose),
                             n, m, k,
                             &alpha,
                             b, t, b_ld,
                             a, t, a_ld,
                             &beta,
                             c, t, c_ld,
                             t, cublasGemmAlgo_t::CUBLAS_GEMM_DFALT);
            } else {
                cublasGemmEx(h, CudaOp(a_transpose), CudaOp(b_transpose),
                             m, n, k,
                             &alpha,
                             a, t, a_ld,
                             b, t, b_ld,
                             &beta,
                             c, t, c_ld,
                             t, cublasGemmAlgo_t::CUBLAS_GEMM_DFALT);
            }
        ));
}

template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const float,
                              const Buffer<float>&, const size_t, const size_t,
                              const Buffer<float>&, const size_t, const size_t,
                              const float,
                              Buffer<float>&, const size_t, const size_t,
                              const Queue&, Event*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const double,
                              const Buffer<double>&, const size_t, const size_t,
                              const Buffer<double>&, const size_t, const size_t,
                              const double,
                              Buffer<double>&, const size_t, const size_t,
                              const Queue&, Event*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const float2,
                              const Buffer<float2>&, const size_t, const size_t,
                              const Buffer<float2>&, const size_t, const size_t,
                              const float2,
                              Buffer<float2>&, const size_t, const size_t,
                              const Queue&, Event*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const double2,
                              const Buffer<double2>&, const size_t, const size_t,
                              const Buffer<double2>&, const size_t, const size_t,
                              const double2,
                              Buffer<double2>&, const size_t, const size_t,
                              const Queue&, Event*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const half,
                              const Buffer<half>&, const size_t, const size_t,
                              const Buffer<half>&, const size_t, const size_t,
                              const half,
                              Buffer<half>&, const size_t, const size_t,
                              const Queue&, Event*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const int32_t,
                              const Buffer<int32_t>&, const size_t, const size_t,
                              const Buffer<int32_t>&, const size_t, const size_t,
                              const int32_t,
                              Buffer<int32_t>&, const size_t, const size_t,
                              const Queue&, Event*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const int64_t,
                              const Buffer<int64_t>&, const size_t, const size_t,
                              const Buffer<int64_t>&, const size_t, const size_t,
                              const int64_t,
                              Buffer<int64_t>&, const size_t, const size_t,
                              const Queue&, Event*);

//---------------------------------------------------------------------------
// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM

template <typename T>
void symm(const Layout layout, const Side side, const Triangle triangle,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xsymm<T>(queue, event);
            routine.DoSymm(layout, side, triangle,
                           m, n,
                           alpha,
                           a_buffer, a_offset, a_ld,
                           b_buffer, b_offset, b_ld,
                           beta,
                           c_buffer, c_offset, c_ld);
        ),
        CUBLAS(
            if (layout == Layout::RowMajor) {
                auto cu_side = side == Side::Left
                    ? cublasSideMode_t::CUBLAS_SIDE_RIGHT
                    : cublasSideMode_t::CUBLAS_SIDE_LEFT;
                auto cu_uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
                cublasSymmEx(h, cu_side, cu_uplo, n, m,
                             &alpha,
                             cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                             cuBuffer::unwrap(b_buffer) + b_offset, b_ld,
                             &beta,
                             cuBuffer::unwrap(c_buffer) + c_offset, c_ld);
            } else {
                auto cu_side = side == Side::Left
                    ? cublasSideMode_t::CUBLAS_SIDE_LEFT
                    : cublasSideMode_t::CUBLAS_SIDE_RIGHT;
                auto cu_uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
                cublasSymmEx(h, cu_side, cu_uplo, m, n,
                             &alpha,
                             cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                             cuBuffer::unwrap(b_buffer) + b_offset, b_ld,
                             &beta,
                             cuBuffer::unwrap(c_buffer) + c_offset, c_ld);
            }
        ));
}

template void PUBLIC_API symm<float>  (const Layout, const Side, const Triangle,
                                       const size_t, const size_t,
                                       const float,
                                       const Buffer<float>&, const size_t, const size_t,
                                       const Buffer<float>&, const size_t, const size_t,
                                       const float,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API symm<double> (const Layout, const Side, const Triangle,
                                       const size_t, const size_t,
                                       const double,
                                       const Buffer<double>&, const size_t, const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       const double,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API symm<float2> (const Layout, const Side, const Triangle,
                                       const size_t, const size_t,
                                       const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const float2,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API symm<double2>(const Layout, const Side, const Triangle,
                                       const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API symm<half>   (const Layout, const Side, const Triangle,
                                       const size_t, const size_t,
                                       const half,
                                       const Buffer<half>&, const size_t, const size_t,
                                       const Buffer<half>&, const size_t, const size_t,
                                       const half,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API symm<int32_t>(const Layout, const Side, const Triangle,
                                       const size_t, const size_t,
                                       const int32_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       const int32_t,
                                       Buffer<int32_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API symm<int64_t>(const Layout, const Side, const Triangle,
                                       const size_t, const size_t,
                                       const int64_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       const int64_t,
                                       Buffer<int64_t>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM

template <typename T>
void hemm(const Layout layout, const Side side, const Triangle triangle,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event) {
    auto routine = Xhemm<T>(queue, event);
    routine.DoHemm(layout, side, triangle,
                   m, n,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   b_buffer, b_offset, b_ld,
                   beta,
                   c_buffer, c_offset, c_ld);
}

template void PUBLIC_API hemm<float2> (const Layout, const Side, const Triangle,
                                       const size_t, const size_t,
                                       const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const float2,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API hemm<double2>(const Layout, const Side, const Triangle,
                                       const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK

template <typename T>
void syrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
          const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xsyrk<T>(queue, event);
            routine.DoSyrk(layout, triangle, a_transpose,
                           n, k,
                           alpha,
                           a_buffer, a_offset, a_ld,
                           beta,
                           c_buffer, c_offset, c_ld);
        ),
        CUBLAS(
            if (layout == Layout::RowMajor) {
                auto cu_uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
                auto cu_trans = a_transpose == Transpose::NoTrans
                    ? cublasOperation_t::CUBLAS_OP_T
                    : cublasOperation_t::CUBLAS_OP_N;
                cublasSyrkEx(h, cu_uplo, cu_trans, k, n,
                             &alpha, cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                             &beta, cuBuffer::unwrap(c_buffer) + c_offset, c_ld);
            } else {
                auto cu_uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
                cublasSyrkEx(h, cu_uplo, CudaOp(a_transpose), n, k,
                             &alpha, cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                             &beta, cuBuffer::unwrap(c_buffer) + c_offset, c_ld);
            }
        ));
}

template void PUBLIC_API syrk<float>  (const Layout, const Triangle, const Transpose,
                                       const size_t, const size_t,
                                       const float,
                                       const Buffer<float>&, const size_t, const size_t,
                                       const float,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API syrk<double> (const Layout, const Triangle, const Transpose,
                                       const size_t, const size_t,
                                       const double,
                                       const Buffer<double>&, const size_t, const size_t,
                                       const double,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API syrk<float2> (const Layout, const Triangle, const Transpose,
                                       const size_t, const size_t,
                                       const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       const float2,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API syrk<double2>(const Layout, const Triangle, const Transpose,
                                       const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API syrk<half>   (const Layout, const Triangle, const Transpose,
                                       const size_t, const size_t,
                                       const half,
                                       const Buffer<half>&, const size_t, const size_t,
                                       const half,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Rank-K update of a hermitian matrix: CHERK/ZHERK

template <typename T>
void herk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
          const size_t n, const size_t k,
          const T alpha,
          const Buffer<std::complex<T>>& a_buffer, const size_t a_offset, const size_t a_ld,
          const T beta,
          Buffer<std::complex<T>>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event) {
    auto routine = Xherk<std::complex<T>,T>(queue, event);
    routine.DoHerk(layout, triangle, a_transpose,
                   n, k,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   beta,
                   c_buffer, c_offset, c_ld);
}

template void PUBLIC_API herk<float> (const Layout, const Triangle, const Transpose,
                                      const size_t, const size_t,
                                      const float,
                                      const Buffer<std::complex<float>>&, const size_t, const size_t,
                                      const float,
                                      Buffer<std::complex<float>>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API herk<double>(const Layout, const Triangle, const Transpose,
                                      const size_t, const size_t,
                                      const double,
                                      const Buffer<std::complex<double>>&, const size_t, const size_t,
                                      const double,
                                      Buffer<std::complex<double>>&, const size_t, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K

template <typename T>
void syr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
           const size_t n, const size_t k,
           const T alpha,
           const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
           const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
           const T beta,
           Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
           const Queue& queue, Event* event) {
    auto routine = Xsyr2k<T>(queue, event);
    routine.DoSyr2k(layout, triangle, ab_transpose,
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    beta,
                    c_buffer, c_offset, c_ld);
}

template void PUBLIC_API syr2k<float>  (const Layout, const Triangle, const Transpose,
                                        const size_t, const size_t,
                                        const float,
                                        const Buffer<float>&, const size_t, const size_t,
                                        const Buffer<float>&, const size_t, const size_t,
                                        const float,
                                        Buffer<float>&, const size_t, const size_t,
                                        const Queue&, Event*);
template void PUBLIC_API syr2k<double> (const Layout, const Triangle, const Transpose,
                                        const size_t, const size_t,
                                        const double,
                                        const Buffer<double>&, const size_t, const size_t,
                                        const Buffer<double>&, const size_t, const size_t,
                                        const double,
                                        Buffer<double>&, const size_t, const size_t,
                                        const Queue&, Event*);
template void PUBLIC_API syr2k<float2> (const Layout, const Triangle, const Transpose,
                                        const size_t, const size_t,
                                        const float2,
                                        const Buffer<float2>&, const size_t, const size_t,
                                        const Buffer<float2>&, const size_t, const size_t,
                                        const float2,
                                        Buffer<float2>&, const size_t, const size_t,
                                        const Queue&, Event*);
template void PUBLIC_API syr2k<double2>(const Layout, const Triangle, const Transpose,
                                        const size_t, const size_t,
                                        const double2,
                                        const Buffer<double2>&, const size_t, const size_t,
                                        const Buffer<double2>&, const size_t, const size_t,
                                        const double2,
                                        Buffer<double2>&, const size_t, const size_t,
                                        const Queue&, Event*);
template void PUBLIC_API syr2k<half>   (const Layout, const Triangle, const Transpose,
                                        const size_t, const size_t,
                                        const half,
                                        const Buffer<half>&, const size_t, const size_t,
                                        const Buffer<half>&, const size_t, const size_t,
                                        const half,
                                        Buffer<half>&, const size_t, const size_t,
                                        const Queue&, Event*);

//---------------------------------------------------------------------------
// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K

template <typename T, typename U>
void her2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
           const size_t n, const size_t k,
           const T alpha,
           const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
           const Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
           const U beta,
           Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
           const Queue& queue, Event* event) {
    auto routine = Xher2k<T,U>(queue, event);
    routine.DoHer2k(layout, triangle, ab_transpose,
                    n, k,
                    alpha,
                    a_buffer, a_offset, a_ld,
                    b_buffer, b_offset, b_ld,
                    beta,
                    c_buffer, c_offset, c_ld);
}

template void PUBLIC_API her2k<float2,float>  (const Layout, const Triangle, const Transpose,
                                               const size_t, const size_t,
                                               const float2,
                                               const Buffer<float2>&, const size_t, const size_t,
                                               const Buffer<float2>&, const size_t, const size_t,
                                               const float,
                                               Buffer<float2>&, const size_t, const size_t,
                                               const Queue&, Event*);
template void PUBLIC_API her2k<double2,double>(const Layout, const Triangle, const Transpose,
                                               const size_t, const size_t,
                                               const double2,
                                               const Buffer<double2>&, const size_t, const size_t,
                                               const Buffer<double2>&, const size_t, const size_t,
                                               const double,
                                               Buffer<double2>&, const size_t, const size_t,
                                               const Queue&, Event*);

//---------------------------------------------------------------------------
// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM

template <typename T>
void trmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t m, const size_t n, const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xtrmm<T>(queue, event);
            routine.DoTrmm(layout, side, triangle, a_transpose, diagonal,
                           m, n, alpha,
                           a_buffer, a_offset, a_ld,
                           b_buffer, b_offset, b_ld);
        ),
        CUBLAS(
            if (layout == Layout::RowMajor) {
                auto cu_side = side == Side::Left
                    ? cublasSideMode_t::CUBLAS_SIDE_RIGHT
                    : cublasSideMode_t::CUBLAS_SIDE_LEFT;
                auto cu_uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
                auto cu_diag = diagonal == Diagonal::Unit
                    ? cublasDiagType_t::CUBLAS_DIAG_UNIT
                    : cublasDiagType_t::CUBLAS_DIAG_NON_UNIT;
                cublasTrmmEx(h, cu_side, cu_uplo, CudaOp(a_transpose), cu_diag,
                             n, m, &alpha,
                             cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                             cuBuffer::unwrap(b_buffer) + b_offset, b_ld,
                             cuBuffer::unwrap(b_buffer) + b_offset, b_ld);
            } else {
                auto cu_side = side == Side::Left
                    ? cublasSideMode_t::CUBLAS_SIDE_LEFT
                    : cublasSideMode_t::CUBLAS_SIDE_RIGHT;
                auto cu_uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
                auto cu_diag = diagonal == Diagonal::Unit
                    ? cublasDiagType_t::CUBLAS_DIAG_UNIT
                    : cublasDiagType_t::CUBLAS_DIAG_NON_UNIT;
                cublasTrmmEx(h, cu_side, cu_uplo, CudaOp(a_transpose), cu_diag,
                             m, n, &alpha,
                             cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                             cuBuffer::unwrap(b_buffer) + b_offset, b_ld,
                             cuBuffer::unwrap(b_buffer) + b_offset, b_ld);
            }
        ));
}

template void PUBLIC_API trmm<float>  (const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const float,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmm<double> (const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const double,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmm<float2> (const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmm<half>   (const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const half,
                                       const Buffer<half>&, const size_t, const size_t,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmm<int32_t>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const int32_t,
                                       const Buffer<int32_t>&, const size_t, const size_t,
                                       Buffer<int32_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmm<int64_t>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const int64_t,
                                       const Buffer<int64_t>&, const size_t, const size_t,
                                       Buffer<int64_t>&, const size_t, const size_t,
                                       const Queue&, Event*);

//---------------------------------------------------------------------------
// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM

template <typename T>
void trsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t m, const size_t n, const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xtrsm<T>(queue, event);
            routine.DoTrsm(layout, side, triangle, a_transpose, diagonal,
                           m, n, alpha,
                           a_buffer, a_offset, a_ld,
                           b_buffer, b_offset, b_ld);
        ),
        CUBLAS(
            if (layout == Layout::RowMajor) {
                auto cu_side = side == Side::Left
                    ? cublasSideMode_t::CUBLAS_SIDE_RIGHT
                    : cublasSideMode_t::CUBLAS_SIDE_LEFT;
                auto cu_uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
                auto cu_diag = diagonal == Diagonal::Unit
                    ? cublasDiagType_t::CUBLAS_DIAG_UNIT
                    : cublasDiagType_t::CUBLAS_DIAG_NON_UNIT;
                cublasTrsmEx(h, cu_side, cu_uplo, CudaOp(a_transpose), cu_diag,
                             n, m, &alpha,
                             cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                             cuBuffer::unwrap(b_buffer) + b_offset, b_ld);
            } else {
                auto cu_side = side == Side::Left
                    ? cublasSideMode_t::CUBLAS_SIDE_LEFT
                    : cublasSideMode_t::CUBLAS_SIDE_RIGHT;
                auto cu_uplo = triangle == Triangle::Lower
                    ? cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                    : cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
                auto cu_diag = diagonal == Diagonal::Unit
                    ? cublasDiagType_t::CUBLAS_DIAG_UNIT
                    : cublasDiagType_t::CUBLAS_DIAG_NON_UNIT;
                cublasTrsmEx(h, cu_side, cu_uplo, CudaOp(a_transpose), cu_diag,
                             m, n, &alpha,
                             cuBuffer::unwrap(a_buffer) + a_offset, a_ld,
                             cuBuffer::unwrap(b_buffer) + b_offset, b_ld);
            }
        ));
}

template void PUBLIC_API trsm<float>  (const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const float,
                                       const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trsm<double> (const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const double,
                                       const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trsm<float2> (const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const float2,
                                       const Buffer<float2>&, const size_t, const size_t,
                                       Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trsm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t, const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

//---------------------------------------------------------------------------
// Element-wise vector product (Hadamard): SHAD/DHAD/CHAD/ZHAD/HHAD

template <typename T>
void had(const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         const T beta,
         Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc,
         const Queue& queue, Event* event) {
    auto routine = Xhad<T>(queue, event);
    routine.DoHad(n,
                  alpha,
                  x_buffer, x_offset, x_inc,
                  y_buffer, y_offset, y_inc,
                  beta,
                  z_buffer, z_offset, z_inc);
}

template void PUBLIC_API had<float>  (const size_t,
                                      const float,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const float,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API had<double> (const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API had<float2> (const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const float2,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API had<double2>(const size_t,
                                      const double2,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      const double2,
                                      Buffer<double2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API had<half>   (const size_t,
                                      const half,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const Buffer<half>&, const size_t, const size_t,
                                      const half,
                                      Buffer<half>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API had<int32_t>(const size_t,
                                      const int32_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      const Buffer<int32_t>&, const size_t, const size_t,
                                      const int32_t,
                                      Buffer<int32_t>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API had<int64_t>(const size_t,
                                      const int64_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      const Buffer<int64_t>&, const size_t, const size_t,
                                      const int64_t,
                                      Buffer<int64_t>&, const size_t, const size_t,
                                      const Queue&, Event*);

//---------------------------------------------------------------------------
// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY

template <typename T>
void omatcopy(const Layout layout, const Transpose a_transpose,
              const size_t m, const size_t n, const T alpha,
              const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
              Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
              const Queue& queue, Event* event) {
    auto routine = Xomatcopy<T>(queue, event);
    routine.DoOmatcopy(layout, a_transpose, m, n, alpha,
                       a_buffer, a_offset, a_ld,
                       b_buffer, b_offset, b_ld);
}

template void PUBLIC_API omatcopy<float>  (const Layout, const Transpose,
                                           const size_t, const size_t, const float,
                                           const Buffer<float>&, const size_t, const size_t,
                                           Buffer<float>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API omatcopy<double> (const Layout, const Transpose,
                                           const size_t, const size_t, const double,
                                           const Buffer<double>&, const size_t, const size_t,
                                           Buffer<double>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API omatcopy<float2> (const Layout, const Transpose,
                                           const size_t, const size_t, const float2,
                                           const Buffer<float2>&, const size_t, const size_t,
                                           Buffer<float2>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API omatcopy<double2>(const Layout, const Transpose,
                                           const size_t, const size_t, const double2,
                                           const Buffer<double2>&, const size_t, const size_t,
                                           Buffer<double2>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API omatcopy<half>   (const Layout, const Transpose,
                                           const size_t, const size_t, const half,
                                           const Buffer<half>&, const size_t, const size_t,
                                           Buffer<half>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API omatcopy<int32_t>(const Layout, const Transpose,
                                           const size_t, const size_t, const int32_t,
                                           const Buffer<int32_t>&, const size_t, const size_t,
                                           Buffer<int32_t>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API omatcopy<int64_t>(const Layout, const Transpose,
                                           const size_t, const size_t, const int64_t,
                                           const Buffer<int64_t>&, const size_t, const size_t,
                                           Buffer<int64_t>&, const size_t, const size_t,
                                           const Queue&, Event*);

//---------------------------------------------------------------------------
// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL

template <typename T>
void im2col(const KernelMode kernel_mode,
            const size_t batches, const size_t channels,
            const size_t height, const size_t width,
            const size_t kernel_h, const size_t kernel_w,
            const size_t pad_t, const size_t pad_l, const size_t pad_b, const size_t pad_r,
            const size_t stride_h, const size_t stride_w,
            const size_t dilation_h, const size_t dilation_w,
            const Buffer<T>& im_buffer, const size_t im_offset,
            Buffer<T>& col_buffer, const size_t col_offset,
            const Queue& queue, Event* event) {
    auto routine = Xim2col<T>(queue, event);
    routine.DoIm2col(kernel_mode,
                     batches, channels,
                     height, width,
                     kernel_h, kernel_w,
                     pad_t, pad_l, pad_b, pad_r,
                     stride_h, stride_w,
                     dilation_h, dilation_w,
                     im_buffer, im_offset,
                     col_buffer, col_offset);
}

template void PUBLIC_API im2col<float>  (const KernelMode,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<float>&, const size_t,
                                         Buffer<float>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API im2col<double> (const KernelMode,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<double>&, const size_t,
                                         Buffer<double>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API im2col<float2> (const KernelMode,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<float2>&, const size_t,
                                         Buffer<float2>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API im2col<double2>(const KernelMode,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<double2>&, const size_t,
                                         Buffer<double2>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API im2col<half>   (const KernelMode,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<half>&, const size_t,
                                         Buffer<half>&, const size_t,
                                         const Queue&, Event*);

//---------------------------------------------------------------------------
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
            const Queue& queue, Event* event) {
    auto routine = Xcol2im<T>(queue, event);
    routine.DoCol2im(kernel_mode,
                     channels, height, width,
                     kernel_h, kernel_w,
                     pad_h, pad_w,
                     stride_h, stride_w,
                     dilation_h, dilation_w,
                     col_buffer, col_offset,
                     im_buffer, im_offset);
}

template void PUBLIC_API col2im<float>  (const KernelMode,
                                         const size_t, const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<float>&, const size_t,
                                         Buffer<float>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API col2im<double> (const KernelMode,
                                         const size_t, const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<double>&, const size_t,
                                         Buffer<double>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API col2im<float2> (const KernelMode,
                                         const size_t, const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<float2>&, const size_t,
                                         Buffer<float2>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API col2im<double2>(const KernelMode,
                                         const size_t, const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<double2>&, const size_t,
                                         Buffer<double2>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API col2im<half>   (const KernelMode,
                                         const size_t, const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<half>&, const size_t,
                                         Buffer<half>&, const size_t,
                                         const Queue&, Event*);

//---------------------------------------------------------------------------
// Batched convolution as GEMM (non-BLAS function): SCONVGEMM/DCONVGEMM/HCONVGEMM

template <typename T>
void convgemm(const KernelMode kernel_mode,
              const size_t batch_count, const size_t channels,
              const size_t height, const size_t width,
              const size_t output_h, const size_t output_w,
              const size_t num_kernels, const size_t kernel_h, const size_t kernel_w,
              const size_t pad_h, const size_t pad_w,
              const size_t stride_h, const size_t stride_w,
              const size_t dilation_h, const size_t dilation_w,
              const Buffer<T>& im_buffer, const size_t im_offset,
              const Buffer<T>& kernel_buffer, const size_t kernel_offset,
              Buffer<T>& result_buffer, const size_t result_offset,
              const Queue& queue, Event* event) {
    auto routine = Xconvgemm<T>(queue, event);
    routine.DoConvgemm(kernel_mode, batch_count, channels,
                       height, width, output_h, output_w,
                       num_kernels, kernel_h, kernel_w,
                       pad_h, pad_w, stride_h, stride_w,
                       dilation_h, dilation_w,
                       im_buffer, im_offset,
                       kernel_buffer, kernel_offset,
                       result_buffer, result_offset);
}

template void PUBLIC_API convgemm<float> (const KernelMode,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const Buffer<float>&, const size_t,
                                          const Buffer<float>&, const size_t,
                                          Buffer<float>&, const size_t,
                                          const Queue&, Event*);
template void PUBLIC_API convgemm<double>(const KernelMode,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const Buffer<double>&, const size_t,
                                          const Buffer<double>&, const size_t,
                                          Buffer<double>&, const size_t,
                                          const Queue&, Event*);
template void PUBLIC_API convgemm<half>  (const KernelMode,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const Buffer<half>&, const size_t,
                                          const Buffer<half>&, const size_t,
                                          Buffer<half>&, const size_t,
                                          const Queue&, Event*);

//---------------------------------------------------------------------------
// Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED

template <typename T>
void axpyBatched(const size_t n, const T* alphas,
                 const Buffer<T>& x_buffer, const size_t* x_offsets, const size_t x_inc,
                 Buffer<T>& y_buffer, const size_t* y_offsets, const size_t y_inc,
                 const size_t batch_count,
                 const Queue& queue, Event* event) {
    auto routine = XaxpyBatched<T>(queue, event);
    auto alphas_cpp = std::vector<T>();
    auto x_offsets_cpp = std::vector<size_t>();
    auto y_offsets_cpp = std::vector<size_t>();
    for (auto batch = size_t{0}; batch < batch_count; ++batch) {
      alphas_cpp.push_back(alphas[batch]);
      x_offsets_cpp.push_back(x_offsets[batch]);
      y_offsets_cpp.push_back(y_offsets[batch]);
    }
    routine.DoAxpyBatched(n, alphas_cpp,
                          x_buffer, x_offsets_cpp, x_inc,
                          y_buffer, y_offsets_cpp, y_inc,
                          batch_count);
}

template void PUBLIC_API axpyBatched<float>  (const size_t, const float*,
                                              const Buffer<float>&, const size_t*, const size_t,
                                              Buffer<float>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API axpyBatched<double> (const size_t, const double*,
                                              const Buffer<double>&, const size_t*, const size_t,
                                              Buffer<double>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API axpyBatched<float2> (const size_t, const float2*,
                                              const Buffer<float2>&, const size_t*, const size_t,
                                              Buffer<float2>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API axpyBatched<double2>(const size_t, const double2*,
                                              const Buffer<double2>&, const size_t*, const size_t,
                                              Buffer<double2>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API axpyBatched<half>   (const size_t, const half*,
                                              const Buffer<half>&, const size_t*, const size_t,
                                              Buffer<half>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);

//---------------------------------------------------------------------------
// Batched version of GEMM: SGEMMBATCHED/DGEMMBATCHED/CGEMMBATCHED/ZGEMMBATCHED/HGEMMBATCHED

template <typename T>
void gemmBatched(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                 const size_t m, const size_t n, const size_t k,
                 const T* alphas,
                 const Buffer<T>& a_buffer, const size_t* a_offsets, const size_t a_ld,
                 const Buffer<T>& b_buffer, const size_t* b_offsets, const size_t b_ld,
                 const T* betas,
                 Buffer<T>& c_buffer, const size_t* c_offsets, const size_t c_ld,
                 const size_t batch_count,
                 const Queue& queue, Event* event)
{
    dispatch<T>(queue,
        OPENCL(
            auto alphas_cpp = std::vector<T>();
            auto betas_cpp = std::vector<T>();
            auto a_offsets_cpp = std::vector<size_t>();
            auto b_offsets_cpp = std::vector<size_t>();
            auto c_offsets_cpp = std::vector<size_t>();

            for (auto batch = size_t{0}; batch < batch_count; ++batch) {
                alphas_cpp.push_back(alphas[batch]);
                betas_cpp.push_back(betas[batch]);
                a_offsets_cpp.push_back(a_offsets[batch]);
                b_offsets_cpp.push_back(b_offsets[batch]);
                c_offsets_cpp.push_back(c_offsets[batch]);
            }

            auto routine = XgemmBatched<T>(queue, event);
            routine.DoGemmBatched(layout, a_transpose, b_transpose,
                                  m, n, k,
                                  alphas_cpp,
                                  a_buffer, a_offsets_cpp, a_ld,
                                  b_buffer, b_offsets_cpp, b_ld,
                                  betas_cpp,
                                  c_buffer, c_offsets_cpp, c_ld,
                                  batch_count);
        ),
        CUBLAS(
            auto t = CudaDataType<T>;

            auto A = cuBuffer::unwrap(a_buffer);
            auto B = cuBuffer::unwrap(b_buffer);
            auto C = cuBuffer::unwrap(c_buffer);

            auto Aarray = std::vector<const T*>();
            auto Barray = std::vector<const T*>();
            auto Carray = std::vector<T*>();

            for (size_t batch = 0; batch < batch_count; ++batch) {
                Aarray.push_back(A + a_offsets[batch]);
                Barray.push_back(B + b_offsets[batch]);
                Carray.push_back(C + c_offsets[batch]);
            }

            auto AarrayBuffer = queue.context().getTemporaryBuffer<const T*>(batch_count);
            auto BarrayBuffer = queue.context().getTemporaryBuffer<const T*>(batch_count);
            auto CarrayBuffer = queue.context().getTemporaryBuffer<T*>(batch_count);

            AarrayBuffer.write(queue, Aarray.data(), batch_count, AarrayBuffer.offset());
            BarrayBuffer.write(queue, Barray.data(), batch_count, BarrayBuffer.offset());
            CarrayBuffer.write(queue, Carray.data(), batch_count, CarrayBuffer.offset());

            if (layout == Layout::RowMajor) {
                cublasGemmBatched(
                    h, CudaOp(b_transpose), CudaOp(a_transpose),
                    n, m, k,
                    alphas,
                    cuBuffer::unwrap(BarrayBuffer), t, b_ld,
                    cuBuffer::unwrap(AarrayBuffer), t, a_ld,
                    betas,
                    cuBuffer::unwrap(CarrayBuffer), t, c_ld,
                    batch_count, t, cublasGemmAlgo_t::CUBLAS_GEMM_DFALT);
            } else {
                cublasGemmBatched(
                    h, CudaOp(a_transpose), CudaOp(b_transpose),
                    m, n, k,
                    alphas,
                    cuBuffer::unwrap(AarrayBuffer), t, a_ld,
                    cuBuffer::unwrap(BarrayBuffer), t, b_ld,
                    betas,
                    cuBuffer::unwrap(CarrayBuffer), t, c_ld,
                    batch_count, t, cublasGemmAlgo_t::CUBLAS_GEMM_DFALT);
            }
        ));
}

template void PUBLIC_API gemmBatched<int16_t>(const Layout, const Transpose, const Transpose,
                                              const size_t, const size_t, const size_t,
                                              const int16_t*,
                                              const Buffer<int16_t>&, const size_t*, const size_t,
                                              const Buffer<int16_t>&, const size_t*, const size_t,
                                              const int16_t*,
                                              Buffer<int16_t>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API gemmBatched<int32_t>(const Layout, const Transpose, const Transpose,
                                              const size_t, const size_t, const size_t,
                                              const int32_t*,
                                              const Buffer<int32_t>&, const size_t*, const size_t,
                                              const Buffer<int32_t>&, const size_t*, const size_t,
                                              const int32_t*,
                                              Buffer<int32_t>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API gemmBatched<int64_t>(const Layout, const Transpose, const Transpose,
                                              const size_t, const size_t, const size_t,
                                              const int64_t*,
                                              const Buffer<int64_t>&, const size_t*, const size_t,
                                              const Buffer<int64_t>&, const size_t*, const size_t,
                                              const int64_t*,
                                              Buffer<int64_t>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API gemmBatched<float>  (const Layout, const Transpose, const Transpose,
                                              const size_t, const size_t, const size_t,
                                              const float*,
                                              const Buffer<float>&, const size_t*, const size_t,
                                              const Buffer<float>&, const size_t*, const size_t,
                                              const float*,
                                              Buffer<float>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API gemmBatched<double> (const Layout, const Transpose, const Transpose,
                                              const size_t, const size_t, const size_t,
                                              const double*,
                                              const Buffer<double>&, const size_t*, const size_t,
                                              const Buffer<double>&, const size_t*, const size_t,
                                              const double*,
                                              Buffer<double>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API gemmBatched<float2> (const Layout, const Transpose, const Transpose,
                                              const size_t, const size_t, const size_t,
                                              const float2*,
                                              const Buffer<float2>&, const size_t*, const size_t,
                                              const Buffer<float2>&, const size_t*, const size_t,
                                              const float2*,
                                              Buffer<float2>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API gemmBatched<double2>(const Layout, const Transpose, const Transpose,
                                              const size_t, const size_t, const size_t,
                                              const double2*,
                                              const Buffer<double2>&, const size_t*, const size_t,
                                              const Buffer<double2>&, const size_t*, const size_t,
                                              const double2*,
                                              Buffer<double2>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API gemmBatched<half>   (const Layout, const Transpose, const Transpose,
                                              const size_t, const size_t, const size_t,
                                              const half*,
                                              const Buffer<half>&, const size_t*, const size_t,
                                              const Buffer<half>&, const size_t*, const size_t,
                                              const half*,
                                              Buffer<half>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);

//---------------------------------------------------------------------------
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
                        const Queue& queue, Event* event) {
    dispatch<T>(queue,
        OPENCL(
            auto routine = XgemmStridedBatched<T>(queue, event);
            routine.DoGemmStridedBatched(
                layout, a_transpose, b_transpose,
                m, n, k,
                alpha,
                a_buffer, a_offset, a_ld, a_stride,
                b_buffer, b_offset, b_ld, b_stride,
                beta,
                c_buffer, c_offset, c_ld, c_stride,
                batch_count);
        ),
        CUBLAS(
            auto t = CudaDataType<T>;
            auto a = cuBuffer::unwrap(a_buffer) + a_offset;
            auto b = cuBuffer::unwrap(b_buffer) + b_offset;
            auto c = cuBuffer::unwrap(c_buffer) + c_offset;

            if (layout == Layout::RowMajor) {
                cublasGemmStridedBatchedEx(
                    h, CudaOp(b_transpose), CudaOp(a_transpose),
                    n, m, k,
                    &alpha,
                    b, t, b_ld, b_stride,
                    a, t, a_ld, a_stride,
                    &beta,
                    c, t, c_ld, c_stride, batch_count,
                    t, cublasGemmAlgo_t::CUBLAS_GEMM_DFALT);
            } else {
                cublasGemmStridedBatchedEx(
                    h, CudaOp(a_transpose), CudaOp(b_transpose),
                    m, n, k,
                    &alpha,
                    a, t, a_ld, a_stride,
                    b, t, b_ld, b_stride,
                    &beta,
                    c, t, c_ld, c_stride, batch_count,
                    t, cublasGemmAlgo_t::CUBLAS_GEMM_DFALT);
            }
        ));
}

template void PUBLIC_API gemmStridedBatched<int16_t>(const Layout, const Transpose, const Transpose,
                                                     const size_t, const size_t, const size_t,
                                                     const int16_t,
                                                     const Buffer<int16_t>&, const size_t, const size_t, const size_t,
                                                     const Buffer<int16_t>&, const size_t, const size_t, const size_t,
                                                     const int16_t,
                                                     Buffer<int16_t>&, const size_t, const size_t, const size_t,
                                                     const size_t,
                                                     const Queue&, Event*);
template void PUBLIC_API gemmStridedBatched<int32_t>(const Layout, const Transpose, const Transpose,
                                                     const size_t, const size_t, const size_t,
                                                     const int32_t,
                                                     const Buffer<int32_t>&, const size_t, const size_t, const size_t,
                                                     const Buffer<int32_t>&, const size_t, const size_t, const size_t,
                                                     const int32_t,
                                                     Buffer<int32_t>&, const size_t, const size_t, const size_t,
                                                     const size_t,
                                                     const Queue&, Event*);
template void PUBLIC_API gemmStridedBatched<int64_t>(const Layout, const Transpose, const Transpose,
                                                     const size_t, const size_t, const size_t,
                                                     const int64_t,
                                                     const Buffer<int64_t>&, const size_t, const size_t, const size_t,
                                                     const Buffer<int64_t>&, const size_t, const size_t, const size_t,
                                                     const int64_t,
                                                     Buffer<int64_t>&, const size_t, const size_t, const size_t,
                                                     const size_t,
                                                     const Queue&, Event*);
template void PUBLIC_API gemmStridedBatched<float>  (const Layout, const Transpose, const Transpose,
                                                     const size_t, const size_t, const size_t,
                                                     const float,
                                                     const Buffer<float>&, const size_t, const size_t, const size_t,
                                                     const Buffer<float>&, const size_t, const size_t, const size_t,
                                                     const float,
                                                     Buffer<float>&, const size_t, const size_t, const size_t,
                                                     const size_t,
                                                     const Queue&, Event*);
template void PUBLIC_API gemmStridedBatched<double> (const Layout, const Transpose, const Transpose,
                                                     const size_t, const size_t, const size_t,
                                                     const double,
                                                     const Buffer<double>&, const size_t, const size_t, const size_t,
                                                     const Buffer<double>&, const size_t, const size_t, const size_t,
                                                     const double,
                                                     Buffer<double>&, const size_t, const size_t, const size_t,
                                                     const size_t,
                                                     const Queue&, Event*);
template void PUBLIC_API gemmStridedBatched<float2> (const Layout, const Transpose, const Transpose,
                                                     const size_t, const size_t, const size_t,
                                                     const float2,
                                                     const Buffer<float2>&, const size_t, const size_t, const size_t,
                                                     const Buffer<float2>&, const size_t, const size_t, const size_t,
                                                     const float2,
                                                     Buffer<float2>&, const size_t, const size_t, const size_t,
                                                     const size_t,
                                                     const Queue&, Event*);
template void PUBLIC_API gemmStridedBatched<double2>(const Layout, const Transpose, const Transpose,
                                                     const size_t, const size_t, const size_t,
                                                     const double2,
                                                     const Buffer<double2>&, const size_t, const size_t, const size_t,
                                                     const Buffer<double2>&, const size_t, const size_t, const size_t,
                                                     const double2,
                                                     Buffer<double2>&, const size_t, const size_t, const size_t,
                                                     const size_t,
                                                     const Queue&, Event*);
template void PUBLIC_API gemmStridedBatched<half>   (const Layout, const Transpose, const Transpose,
                                                     const size_t, const size_t, const size_t,
                                                     const half,
                                                     const Buffer<half>&, const size_t, const size_t, const size_t,
                                                     const Buffer<half>&, const size_t, const size_t, const size_t,
                                                     const half,
                                                     Buffer<half>&, const size_t, const size_t, const size_t,
                                                     const size_t,
                                                     const Queue&, Event*);

// =================================================================================================
// LAPACK routines
// =================================================================================================

template <typename T>
void getrf(const size_t m, const size_t n,
           Buffer<T>& A, const size_t a_offset, const size_t lda,
           Buffer<int32_t>& ipiv, const size_t ipiv_offset,
           const Queue& queue, Event* event)
{
    dispatch<T>(queue,
        OPENCL(
            auto routine = Xgetrf<T>(queue, event);
            routine.DoGetrf(m, n, A, a_offset, lda, ipiv, ipiv_offset);
        ),
        CUSOLVER(
            int lwork = 0;
            cusolverDnXgetrf_bufferSize(
                h, m, n, cuBuffer::unwrap(A) + a_offset, lda, &lwork);
            auto work = queue.context().getTemporaryBuffer<T>(lwork);

            cusolverDnXgetrf(
                h, m, n,
                cuBuffer::unwrap(A) + a_offset, lda,
                cuBuffer::unwrap(work),
                cuBuffer::unwrap(ipiv) + ipiv_offset + 1,
                cuBuffer::unwrap(ipiv) + ipiv_offset);
        ));
}

template void PUBLIC_API getrf<float>(const size_t, const size_t,
                                      Buffer<float>&, const size_t, const size_t,
                                      Buffer<int32_t>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API getrf<double>(const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      Buffer<int32_t>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API getrf<float2>(const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      Buffer<int32_t>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API getrf<double2>(const size_t, const size_t,
                                      Buffer<double2>&, const size_t, const size_t,
                                      Buffer<int32_t>&, const size_t,
                                      const Queue&, Event*);

template <typename T>
void getrs(Transpose trans, const size_t n, const size_t nrhs,
           const Buffer<T>& A, const size_t a_offset, const size_t lda,
           const Buffer<int32_t>& ipiv, const size_t ipiv_offset,
           Buffer<T>& B, const size_t b_offset, const size_t ldb,
           const Queue& queue, Event* event)
{
    if (trans == Transpose::ConjTrans) {
        auto routine = Xgetrf<T>(queue, event);
        routine.DoGetrs(trans, n, nrhs, A, a_offset, lda, ipiv, ipiv_offset, B, b_offset, ldb);
        return;
    }

    dispatch<T>(queue,
        OPENCL(
            auto routine = Xgetrf<T>(queue, event);
            routine.DoGetrs(trans, n, nrhs, A, a_offset, lda, ipiv, ipiv_offset, B, b_offset, ldb);
        ),
        CUSOLVER(
            auto cuop = trans == Transpose::NoTrans
                ? cublasOperation_t::CUBLAS_OP_T
                : cublasOperation_t::CUBLAS_OP_N;
            auto info = queue.context().getTemporaryBuffer<int>(1);

            auto work = queue.context().getTemporaryBuffer<T>(n * nrhs);
            omatcopy(Layout::RowMajor, Transpose::Trans, n, nrhs,
                PrecisionTraits<T>::One, B, b_offset, ldb,
                work, work.offset(), n , queue);

            cusolverDnXgetrs(h, cuop, n, nrhs,
                cuBuffer::unwrap(A) + a_offset, lda,
                cuBuffer::unwrap(ipiv) + ipiv_offset + 1,
                cuBuffer::unwrap(work), n,
                cuBuffer::unwrap(info) + info.offset());

            omatcopy(Layout::RowMajor, Transpose::Trans, nrhs, n,
                PrecisionTraits<T>::One, work, work.offset(), n,
                B, b_offset, ldb, queue);
        ));
}

template void PUBLIC_API getrs<float>(Transpose, const size_t, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const Buffer<int32_t>&, const size_t,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API getrs<double>(Transpose, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<int32_t>&, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API getrs<float2>(Transpose, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<int32_t>&, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API getrs<double2>(Transpose, const size_t, const size_t,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      const Buffer<int32_t>&, const size_t,
                                      Buffer<double2>&, const size_t, const size_t,
                                      const Queue&, Event*);

// =================================================================================================
}} // namespace gpgpu::blas
