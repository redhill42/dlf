
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

#include "routines/routines.hpp"
#include "gpblas.h"

namespace gpgpu::blas {

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
template <typename T>
void rotg(Buffer<T>&, const size_t,
          Buffer<T>&, const size_t,
          Buffer<T>&, const size_t,
          Buffer<T>&, const size_t,
          const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API rotg<float>(Buffer<float>&, const size_t,
                                     Buffer<float>&, const size_t,
                                     Buffer<float>&, const size_t,
                                     Buffer<float>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API rotg<double>(Buffer<double>&, const size_t,
                                      Buffer<double>&, const size_t,
                                      Buffer<double>&, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);

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

template void PUBLIC_API rotmg<float>(Buffer<float>&, const size_t,
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

// Apply givens plane rotation: SROT/DROT
template <typename T>
void rot(const size_t,
         Buffer<T>&, const size_t, const size_t,
         Buffer<T>&, const size_t, const size_t,
         const T,
         const T,
         const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API rot<float>(const size_t,
                                    Buffer<float>&, const size_t, const size_t,
                                    Buffer<float>&, const size_t, const size_t,
                                    const float,
                                    const float,
                                    const Queue&, Event*);
template void PUBLIC_API rot<double>(const size_t,
                                     Buffer<double>&, const size_t, const size_t,
                                     Buffer<double>&, const size_t, const size_t,
                                     const double,
                                     const double,
                                     const Queue&, Event*);

// Apply modified givens plane rotation: SROTM/DROTM
template <typename T>
void rotm(const size_t,
          Buffer<T>&, const size_t, const size_t,
          Buffer<T>&, const size_t, const size_t,
          Buffer<T>&, const size_t,
          const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API rotm<float>(const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API rotm<double>(const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
template <typename T>
void swap(const size_t n,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xswap<T>(queue, event);
    routine.DoSwap(n,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API swap<float>(const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API swap<double>(const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API swap<float2>(const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API swap<double2>(const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API swap<half>(const size_t,
                                    Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
template <typename T>
void scal(const size_t n,
          const T alpha,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event) {
    auto routine = Xscal<T>(queue, event);
    routine.DoScal(n, alpha, x_buffer, x_offset, x_inc);
}

template void PUBLIC_API scal<float>(const size_t, const float,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API scal<double>(const size_t, const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API scal<float2>(const size_t, const float2,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API scal<double2>(const size_t, const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API scal<half>(const size_t, const half,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
template <typename T>
void copy(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xcopy<T>(queue, event);
    routine.DoCopy(n,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API copy<float>(const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API copy<double>(const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API copy<float2>(const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API copy<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<half>(const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
template <typename T>
void axpy(const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xaxpy<T>(queue, event);
    routine.DoAxpy(n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API axpy<float>(const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API axpy<double>(const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API axpy<float2>(const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API axpy<double2>(const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API axpy<half>(const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Dot product of two vectors: SDOT/DDOT/HDOT
template <typename T>
void dot(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& dot_buffer, const size_t dot_offset,
         const Queue& queue, Event* event) {
    auto routine = Xdot<T>(queue, event);
    routine.DoDot(n,
                  x_buffer, x_offset, x_inc,
                  y_buffer, y_offset, y_inc,
                  dot_buffer, dot_offset);
}

template void PUBLIC_API dot<float>(const size_t,
                                    const Buffer<float>&, const size_t, const size_t,
                                    const Buffer<float>&, const size_t, const size_t,
                                    Buffer<float>&, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API dot<double>(const size_t,
                                     const Buffer<double>&, const size_t, const size_t,
                                     const Buffer<double>&, const size_t, const size_t,
                                     Buffer<double>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API dot<half>(const size_t,
                                   const Buffer<half>&, const size_t, const size_t,
                                   const Buffer<half>&, const size_t, const size_t,
                                   Buffer<half>&, const size_t,
                                   const Queue&, Event*);

// Dot product of two complex vectors: CDOTU/ZDOTU
template <typename T>
void dotu(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& dot_buffer, const size_t dot_offset,
          const Queue& queue, Event* event) {
    auto routine = Xdotu<T>(queue, event);
    routine.DoDotu(n,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   dot_buffer, dot_offset);
}

template void PUBLIC_API dotu<float2>(const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API dotu<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
template <typename T>
void dotc(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& dot_buffer, const size_t dot_offset,
          const Queue& queue, Event* event) {
    auto routine = Xdotc<T>(queue, event);
    routine.DoDotc(n,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   dot_buffer, dot_offset);
}

template void PUBLIC_API dotc<float2>(const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API dotc<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
template <typename T>
void nrm2(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& nrm2_buffer, const size_t nrm2_offset,
          const Queue& queue, Event* event) {
    auto routine = Xnrm2<T>(queue, event);
    routine.DoNrm2(n,
                   x_buffer, x_offset, x_inc,
                   nrm2_buffer, nrm2_offset);
}

template void PUBLIC_API nrm2<float>(const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API nrm2<double>(const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API nrm2<float2>(const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API nrm2<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API nrm2<half>(const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t,
                                    const Queue&, Event*);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
template <typename T>
void asum(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& asum_buffer, const size_t asum_offset,
          const Queue& queue, Event* event) {
    auto routine = Xasum<T>(queue, event);
    routine.DoAsum(n,
                   x_buffer, x_offset, x_inc,
                   asum_buffer, asum_offset);
}

template void PUBLIC_API asum<float>(const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API asum<double>(const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API asum<float2>(const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API asum<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API asum<half>(const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t,
                                    const Queue&, Event*);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
template <typename T>
void sum(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& sum_buffer, const size_t sum_offset,
         const Queue& queue, Event* event) {
    auto routine = Xsum<T>(queue, event);
    routine.DoSum(n,
                  x_buffer, x_offset, x_inc,
                  sum_buffer, sum_offset);
}

template void PUBLIC_API sum<float>(const size_t,
                                    const Buffer<float>&, const size_t, const size_t,
                                    Buffer<float>&, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API sum<double>(const size_t,
                                     const Buffer<double>&, const size_t, const size_t,
                                     Buffer<double>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API sum<float2>(const size_t,
                                     const Buffer<float2>&, const size_t, const size_t,
                                     Buffer<float2>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API sum<double2>(const size_t,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      Buffer<double2>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API sum<half>(const size_t,
                                   const Buffer<half>&, const size_t, const size_t,
                                   Buffer<half>&, const size_t,
                                   const Queue&, Event*);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
template <typename T>
void amax(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
          const Queue& queue, Event* event) {
    auto routine = Xamax<T>(queue, event);
    routine.DoAmax(n,
                   x_buffer, x_offset, x_inc,
                   imax_buffer, imax_offset);
}

template void PUBLIC_API amax<float>(const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<unsigned int>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API amax<double>(const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API amax<float2>(const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API amax<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amax<half>(const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<unsigned int>&, const size_t,
                                    const Queue&, Event*);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
template <typename T>
void amin(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<unsigned int>& imin_buffer, const size_t imin_offset,
          const Queue& queue, Event* event) {
    auto routine = Xamin<T>(queue, event);
    routine.DoAmin(n,
                   x_buffer, x_offset, x_inc,
                   imin_buffer, imin_offset);
}

template void PUBLIC_API amin<float>(const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<unsigned int>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API amin<double>(const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API amin<float2>(const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API amin<double2>(const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<unsigned int>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API amin<half>(const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<unsigned int>&, const size_t,
                                    const Queue&, Event*);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
template <typename T>
void max(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<unsigned int>& imax_buffer, const size_t imax_offset,
         const Queue& queue, Event* event) {
    auto routine = Xmax<T>(queue, event);
    routine.DoMax(n,
                  x_buffer, x_offset, x_inc,
                  imax_buffer, imax_offset);
}

template void PUBLIC_API max<float>(const size_t,
                                    const Buffer<float>&, const size_t, const size_t,
                                    Buffer<unsigned int>&, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API max<double>(const size_t,
                                     const Buffer<double>&, const size_t, const size_t,
                                     Buffer<unsigned int>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API max<float2>(const size_t,
                                     const Buffer<float2>&, const size_t, const size_t,
                                     Buffer<unsigned int>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API max<double2>(const size_t,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API max<half>(const size_t,
                                   const Buffer<half>&, const size_t, const size_t,
                                   Buffer<unsigned int>&, const size_t,
                                   const Queue&, Event*);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
template <typename T>
void min(const size_t n,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<unsigned int>& imin_buffer, const size_t imin_offset,
         const Queue& queue, Event* event) {
    auto routine = Xmin<T>(queue, event);
    routine.DoMin(n,
                  x_buffer, x_offset, x_inc,
                  imin_buffer, imin_offset);
}

template void PUBLIC_API min<float>(const size_t,
                                    const Buffer<float>&, const size_t, const size_t,
                                    Buffer<unsigned int>&, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API min<double>(const size_t,
                                     const Buffer<double>&, const size_t, const size_t,
                                     Buffer<unsigned int>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API min<float2>(const size_t,
                                     const Buffer<float2>&, const size_t, const size_t,
                                     Buffer<unsigned int>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API min<double2>(const size_t,
                                      const Buffer<double2>&, const size_t, const size_t,
                                      Buffer<unsigned int>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API min<half>(const size_t,
                                   const Buffer<half>&, const size_t, const size_t,
                                   Buffer<unsigned int>&, const size_t,
                                   const Queue&, Event*);

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
          const Queue& queue, Event* event) {
    auto routine = Xgemv<T>(queue, event);
    routine.DoGemv(layout, a_transpose,
                   m, n,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API gemv<float>(const Layout, const Transpose,
                                     const size_t, const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const float,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API gemv<double>(const Layout, const Transpose,
                                      const size_t, const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API gemv<float2>(const Layout, const Transpose,
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
template void PUBLIC_API gemv<half>(const Layout, const Transpose,
                                    const size_t, const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const half,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

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

template void PUBLIC_API gbmv<float>(const Layout, const Transpose,
                                     const size_t, const size_t, const size_t, const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const float,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API gbmv<double>(const Layout, const Transpose,
                                      const size_t, const size_t, const size_t, const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API gbmv<float2>(const Layout, const Transpose,
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
template void PUBLIC_API gbmv<half>(const Layout, const Transpose,
                                    const size_t, const size_t, const size_t, const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const half,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
template <typename T>
void hemv(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xhemv<T>(queue, event);
    routine.DoHemv(layout, triangle,
                   n,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API hemv<float2>(const Layout, const Triangle,
                                      const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const float2,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API hemv<double2>(const Layout, const Triangle,
                                       const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

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

template void PUBLIC_API hbmv<float2>(const Layout, const Triangle,
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

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
template <typename T>
void hpmv(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& ap_buffer, const size_t ap_offset,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xhpmv<T>(queue, event);
    routine.DoHpmv(layout, triangle,
                   n,
                   alpha,
                   ap_buffer, ap_offset,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API hpmv<float2>(const Layout, const Triangle,
                                      const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const float2,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API hpmv<double2>(const Layout, const Triangle,
                                       const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const double2,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
template <typename T>
void symv(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xsymv<T>(queue, event);
    routine.DoSymv(layout, triangle,
                   n,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API symv<float>(const Layout, const Triangle,
                                     const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const float,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API symv<double>(const Layout, const Triangle,
                                      const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API symv<half>(const Layout, const Triangle,
                                    const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const half,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

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
    routine.DoSbmv(layout, triangle,
                   n, k,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API sbmv<float>(const Layout, const Triangle,
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
template void PUBLIC_API sbmv<half>(const Layout, const Triangle,
                                    const size_t, const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const half,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
template <typename T>
void spmv(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& ap_buffer, const size_t ap_offset,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const T beta,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xspmv<T>(queue, event);
    routine.DoSpmv(layout, triangle,
                   n,
                   alpha,
                   ap_buffer, ap_offset,
                   x_buffer, x_offset, x_inc,
                   beta,
                   y_buffer, y_offset, y_inc);
}

template void PUBLIC_API spmv<float>(const Layout, const Triangle,
                                     const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const float,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API spmv<double>(const Layout, const Triangle,
                                      const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API spmv<half>(const Layout, const Triangle,
                                    const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const half,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
template <typename T>
void trmv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
                const size_t n,
                const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
                Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                const Queue& queue, Event* event) {
    auto routine = Xtrmv<T>(queue, event);
    routine.DoTrmv(layout, triangle, a_transpose, diagonal,
                   n,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc);
}

template void PUBLIC_API trmv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                     const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API trmv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API trmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API trmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmv<half>(const Layout, const Triangle, const Transpose, const Diagonal,
                                    const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

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

template void PUBLIC_API tbmv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                     const size_t, const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API tbmv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API tbmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API tbmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tbmv<half>(const Layout, const Triangle, const Transpose, const Diagonal,
                                    const size_t, const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

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

template void PUBLIC_API tpmv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                     const size_t,
                                     const Buffer<float>&, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API tpmv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t,
                                      const Buffer<double>&, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API tpmv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t,
                                      const Buffer<float2>&, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API tpmv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double2>&, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API tpmv<half>(const Layout, const Triangle, const Transpose, const Diagonal,
                                    const size_t,
                                    const Buffer<half>&, const size_t,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
template <typename T>
void trsv(const Layout layout, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t n,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Queue& queue, Event* event) {
    auto routine = Xtrsv<T>(queue, event);
    routine.DoTrsv(layout, triangle, a_transpose, diagonal,
                   n,
                   a_buffer, a_offset, a_ld,
                   x_buffer, x_offset, x_inc);
}

template void PUBLIC_API trsv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                     const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API trsv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API trsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API trsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
template <typename T>
void tbsv(const Layout, const Triangle, const Transpose, const Diagonal,
          const size_t, const size_t,
          const Buffer<T>&, const size_t, const size_t,
          Buffer<T>&, const size_t, const size_t,
          const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API tbsv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                     const size_t, const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API tbsv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API tbsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API tbsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
template <typename T>
void tpsv(const Layout, const Triangle, const Transpose, const Diagonal,
          const size_t,
          const Buffer<T>&, const size_t,
          Buffer<T>&, const size_t, const size_t,
          const Queue&, Event*) {
  throw BLASError(StatusCode::kNotImplemented);
}

template void PUBLIC_API tpsv<float>(const Layout, const Triangle, const Transpose, const Diagonal,
                                     const size_t,
                                     const Buffer<float>&, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API tpsv<double>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t,
                                      const Buffer<double>&, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API tpsv<float2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                      const size_t,
                                      const Buffer<float2>&, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API tpsv<double2>(const Layout, const Triangle, const Transpose, const Diagonal,
                                       const size_t,
                                       const Buffer<double2>&, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

// General rank-1 matrix update: SGER/DGER/HGER
template <typename T>
void ger(const Layout layout,
         const size_t m, const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
         const Queue& queue, Event* event) {
    auto routine = Xger<T>(queue, event);
    routine.DoGer(layout,
                  m, n,
                  alpha,
                  x_buffer, x_offset, x_inc,
                  y_buffer, y_offset, y_inc,
                  a_buffer, a_offset, a_ld);
}

template void PUBLIC_API ger<float>(const Layout,
                                    const size_t, const size_t,
                                    const float,
                                    const Buffer<float>&, const size_t, const size_t,
                                    const Buffer<float>&, const size_t, const size_t,
                                    Buffer<float>&, const size_t, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API ger<double>(const Layout,
                                     const size_t, const size_t,
                                     const double,
                                     const Buffer<double>&, const size_t, const size_t,
                                     const Buffer<double>&, const size_t, const size_t,
                                     Buffer<double>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API ger<half>(const Layout,
                                   const size_t, const size_t,
                                   const half,
                                   const Buffer<half>&, const size_t, const size_t,
                                   const Buffer<half>&, const size_t, const size_t,
                                   Buffer<half>&, const size_t, const size_t,
                                   const Queue&, Event*);

// General rank-1 complex matrix update: CGERU/ZGERU
template <typename T>
void geru(const Layout layout,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event) {
    auto routine = Xgeru<T>(queue, event);
    routine.DoGeru(layout,
                   m, n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   a_buffer, a_offset, a_ld);
}

template void PUBLIC_API geru<float2>(const Layout,
                                      const size_t, const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API geru<double2>(const Layout,
                                       const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
template <typename T>
void gerc(const Layout layout,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event) {
    auto routine = Xgerc<T>(queue, event);
    routine.DoGerc(layout,
                   m, n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   a_buffer, a_offset, a_ld);
}

template void PUBLIC_API gerc<float2>(const Layout,
                                      const size_t, const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API gerc<double2>(const Layout,
                                       const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

// Hermitian rank-1 matrix update: CHER/ZHER
template <typename T>
void her(const Layout layout, const Triangle triangle,
         const size_t n,
         const T alpha,
         const Buffer<std::complex<T>>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<std::complex<T>>& a_buffer, const size_t a_offset, const size_t a_ld,
         const Queue& queue, Event* event) {
    auto routine = Xher<std::complex<T>,T>(queue, event);
    routine.DoHer(layout, triangle,
                  n,
                  alpha,
                  x_buffer, x_offset, x_inc,
                  a_buffer, a_offset, a_ld);
}

template void PUBLIC_API her<float>(const Layout, const Triangle,
                                    const size_t,
                                    const float,
                                    const Buffer<std::complex<float>>&, const size_t, const size_t,
                                    Buffer<std::complex<float>>&, const size_t, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API her<double>(const Layout, const Triangle,
                                     const size_t,
                                     const double,
                                     const Buffer<std::complex<double>>&, const size_t, const size_t,
                                     Buffer<std::complex<double>>&, const size_t, const size_t,
                                     const Queue&, Event*);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
template <typename T>
void hpr(const Layout layout, const Triangle triangle,
         const size_t n,
         const T alpha,
         const Buffer<std::complex<T>>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<std::complex<T>>& ap_buffer, const size_t ap_offset,
         const Queue& queue, Event* event) {
    auto routine = Xhpr<std::complex<T>,T>(queue, event);
    routine.DoHpr(layout, triangle,
                  n,
                  alpha,
                  x_buffer, x_offset, x_inc,
                  ap_buffer, ap_offset);
}

template void PUBLIC_API hpr<float>(const Layout, const Triangle,
                                    const size_t,
                                    const float,
                                    const Buffer<std::complex<float>>&, const size_t, const size_t,
                                    Buffer<std::complex<float>>&, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API hpr<double>(const Layout, const Triangle,
                                     const size_t,
                                     const double,
                                     const Buffer<std::complex<double>>&, const size_t, const size_t,
                                     Buffer<std::complex<double>>&, const size_t,
                                     const Queue&, Event*);

// Hermitian rank-2 matrix update: CHER2/ZHER2
template <typename T>
void her2(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event) {
    auto routine = Xher2<T>(queue, event);
    routine.DoHer2(layout, triangle,
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   a_buffer, a_offset, a_ld);
}

template void PUBLIC_API her2<float2>(const Layout, const Triangle,
                                      const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API her2<double2>(const Layout, const Triangle,
                                       const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
template <typename T>
void hpr2(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& ap_buffer, const size_t ap_offset,
          const Queue& queue, Event* event) {
    auto routine = Xhpr2<T>(queue, event);
    routine.DoHpr2(layout, triangle,
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   ap_buffer, ap_offset);
}

template void PUBLIC_API hpr2<float2>(const Layout, const Triangle,
                                      const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API hpr2<double2>(const Layout, const Triangle,
                                       const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t,
                                       const Queue&, Event*);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
template <typename T>
void syr(const Layout layout, const Triangle triangle,
         const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
         const Queue& queue, Event* event) {
    auto routine = Xsyr<T>(queue, event);
    routine.DoSyr(layout, triangle,
                  n,
                  alpha,
                  x_buffer, x_offset, x_inc,
                  a_buffer, a_offset, a_ld);
}

template void PUBLIC_API syr<float>(const Layout, const Triangle,
                                    const size_t,
                                    const float,
                                    const Buffer<float>&, const size_t, const size_t,
                                    Buffer<float>&, const size_t, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API syr<double>(const Layout, const Triangle,
                                     const size_t,
                                     const double,
                                     const Buffer<double>&, const size_t, const size_t,
                                     Buffer<double>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API syr<half>(const Layout, const Triangle,
                                   const size_t,
                                   const half,
                                   const Buffer<half>&, const size_t, const size_t,
                                   Buffer<half>&, const size_t, const size_t,
                                   const Queue&, Event*);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
template <typename T>
void spr(const Layout layout, const Triangle triangle,
         const size_t n,
         const T alpha,
         const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& ap_buffer, const size_t ap_offset,
         const Queue& queue, Event* event) {
    auto routine = Xspr<T>(queue, event);
    routine.DoSpr(layout, triangle,
                  n,
                  alpha,
                  x_buffer, x_offset, x_inc,
                  ap_buffer, ap_offset);
}

template void PUBLIC_API spr<float>(const Layout, const Triangle,
                                    const size_t,
                                    const float,
                                    const Buffer<float>&, const size_t, const size_t,
                                    Buffer<float>&, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API spr<double>(const Layout, const Triangle,
                                     const size_t,
                                     const double,
                                     const Buffer<double>&, const size_t, const size_t,
                                     Buffer<double>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API spr<half>(const Layout, const Triangle,
                                   const size_t,
                                   const half,
                                   const Buffer<half>&, const size_t, const size_t,
                                   Buffer<half>&, const size_t,
                                   const Queue&, Event*);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
template <typename T>
void syr2(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const Queue& queue, Event* event) {
    auto routine = Xsyr2<T>(queue, event);
    routine.DoSyr2(layout, triangle,
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   a_buffer, a_offset, a_ld);
}

template void PUBLIC_API syr2<float>(const Layout, const Triangle,
                                     const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API syr2<double>(const Layout, const Triangle,
                                      const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API syr2<half>(const Layout, const Triangle,
                                    const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
template <typename T>
void spr2(const Layout layout, const Triangle triangle,
          const size_t n,
          const T alpha,
          const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          Buffer<T>& ap_buffer, const size_t ap_offset,
          const Queue& queue, Event* event) {
    auto routine = Xspr2<T>(queue, event);
    routine.DoSpr2(layout, triangle,
                   n,
                   alpha,
                   x_buffer, x_offset, x_inc,
                   y_buffer, y_offset, y_inc,
                   ap_buffer, ap_offset);
}

template void PUBLIC_API spr2<float>(const Layout, const Triangle,
                                     const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API spr2<double>(const Layout, const Triangle,
                                      const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API spr2<half>(const Layout, const Triangle,
                                    const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t,
                                    const Queue&, Event*);

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
          const Queue& queue, Event* event,
          Buffer<T>* temp_buffer) {
    auto routine = Xgemm<T>(queue, event);
    routine.DoGemm(layout, a_transpose, b_transpose,
                   m, n, k,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   b_buffer, b_offset, b_ld,
                   beta,
                   c_buffer, c_offset, c_ld,
                   temp_buffer);
}

template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const float,
                              const Buffer<float>&, const size_t, const size_t,
                              const Buffer<float>&, const size_t, const size_t,
                              const float,
                              Buffer<float>&, const size_t, const size_t,
                              const Queue&, Event*, Buffer<float>*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const double,
                              const Buffer<double>&, const size_t, const size_t,
                              const Buffer<double>&, const size_t, const size_t,
                              const double,
                              Buffer<double>&, const size_t, const size_t,
                              const Queue&, Event*, Buffer<double>*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const float2,
                              const Buffer<float2>&, const size_t, const size_t,
                              const Buffer<float2>&, const size_t, const size_t,
                              const float2,
                              Buffer<float2>&, const size_t, const size_t,
                              const Queue&, Event*, Buffer<float2>*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const double2,
                              const Buffer<double2>&, const size_t, const size_t,
                              const Buffer<double2>&, const size_t, const size_t,
                              const double2,
                              Buffer<double2>&, const size_t, const size_t,
                              const Queue&, Event*, Buffer<double2>*);
template void PUBLIC_API gemm(const Layout, const Transpose, const Transpose,
                              const size_t, const size_t, const size_t,
                              const half,
                              const Buffer<half>&, const size_t, const size_t,
                              const Buffer<half>&, const size_t, const size_t,
                              const half,
                              Buffer<half>&, const size_t, const size_t,
                              const Queue&, Event*, Buffer<half>*);

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
    auto routine = Xsymm<T>(queue, event);
    routine.DoSymm(layout, side, triangle,
                   m, n,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   b_buffer, b_offset, b_ld,
                   beta,
                   c_buffer, c_offset, c_ld);
}

template void PUBLIC_API symm<float>(const Layout, const Side, const Triangle,
                                     const size_t, const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const float,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API symm<double>(const Layout, const Side, const Triangle,
                                      const size_t, const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API symm<float2>(const Layout, const Side, const Triangle,
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
template void PUBLIC_API symm<half>(const Layout, const Side, const Triangle,
                                    const size_t, const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const half,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

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

template void PUBLIC_API hemm<float2>(const Layout, const Side, const Triangle,
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

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
template <typename T>
void syrk(const Layout layout, const Triangle triangle, const Transpose a_transpose,
          const size_t n, const size_t k,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          const T beta,
          Buffer<T>& c_buffer, const size_t c_offset, const size_t c_ld,
          const Queue& queue, Event* event) {
    auto routine = Xsyrk<T>(queue, event);
    routine.DoSyrk(layout, triangle, a_transpose,
                   n, k,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   beta,
                   c_buffer, c_offset, c_ld);
}

template void PUBLIC_API syrk<float>(const Layout, const Triangle, const Transpose,
                                     const size_t, const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     const float,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API syrk<double>(const Layout, const Triangle, const Transpose,
                                      const size_t, const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      const double,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API syrk<float2>(const Layout, const Triangle, const Transpose,
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
template void PUBLIC_API syrk<half>(const Layout, const Triangle, const Transpose,
                                    const size_t, const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    const half,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

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

template void PUBLIC_API herk<float>(const Layout, const Triangle, const Transpose,
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

template void PUBLIC_API syr2k<float>(const Layout, const Triangle, const Transpose,
                                      const size_t, const size_t,
                                      const float,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const Buffer<float>&, const size_t, const size_t,
                                      const float,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API syr2k<double>(const Layout, const Triangle, const Transpose,
                                       const size_t, const size_t,
                                       const double,
                                       const Buffer<double>&, const size_t, const size_t,
                                       const Buffer<double>&, const size_t, const size_t,
                                       const double,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API syr2k<float2>(const Layout, const Triangle, const Transpose,
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
template void PUBLIC_API syr2k<half>(const Layout, const Triangle, const Transpose,
                                     const size_t, const size_t,
                                     const half,
                                     const Buffer<half>&, const size_t, const size_t,
                                     const Buffer<half>&, const size_t, const size_t,
                                     const half,
                                     Buffer<half>&, const size_t, const size_t,
                                     const Queue&, Event*);

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

template void PUBLIC_API her2k<float2,float>(const Layout, const Triangle, const Transpose,
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

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
template <typename T>
void trmm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const Queue& queue, Event* event) {
    auto routine = Xtrmm<T>(queue, event);
    routine.DoTrmm(layout, side, triangle, a_transpose, diagonal,
                   m, n,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   b_buffer, b_offset, b_ld);
}

template void PUBLIC_API trmm<float>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                     const size_t, const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API trmm<double>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                      const size_t, const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API trmm<float2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                      const size_t, const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API trmm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API trmm<half>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                    const size_t, const size_t,
                                    const half,
                                    const Buffer<half>&, const size_t, const size_t,
                                    Buffer<half>&, const size_t, const size_t,
                                    const Queue&, Event*);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
template <typename T>
void trsm(const Layout layout, const Side side, const Triangle triangle, const Transpose a_transpose, const Diagonal diagonal,
          const size_t m, const size_t n,
          const T alpha,
          const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
          Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
          const Queue& queue, Event* event) {
    auto routine = Xtrsm<T>(queue, event);
    routine.DoTrsm(layout, side, triangle, a_transpose, diagonal,
                   m, n,
                   alpha,
                   a_buffer, a_offset, a_ld,
                   b_buffer, b_offset, b_ld);
}

template void PUBLIC_API trsm<float>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                     const size_t, const size_t,
                                     const float,
                                     const Buffer<float>&, const size_t, const size_t,
                                     Buffer<float>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API trsm<double>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                      const size_t, const size_t,
                                      const double,
                                      const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API trsm<float2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                      const size_t, const size_t,
                                      const float2,
                                      const Buffer<float2>&, const size_t, const size_t,
                                      Buffer<float2>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API trsm<double2>(const Layout, const Side, const Triangle, const Transpose, const Diagonal,
                                       const size_t, const size_t,
                                       const double2,
                                       const Buffer<double2>&, const size_t, const size_t,
                                       Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);

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
         const Queue& queue, Event* event) {
    auto routine = Xhad<T>(queue, event);
    routine.DoHad(n,
                  alpha,
                  x_buffer, x_offset, x_inc,
                  y_buffer, y_offset, y_inc,
                  beta,
                  z_buffer, z_offset, z_inc);
}

template void PUBLIC_API had<float>(const size_t,
                                    const float,
                                    const Buffer<float>&, const size_t, const size_t,
                                    const Buffer<float>&, const size_t, const size_t,
                                    const float,
                                    Buffer<float>&, const size_t, const size_t,
                                    const Queue&, Event*);
template void PUBLIC_API had<double>(const size_t,
                                     const double,
                                     const Buffer<double>&, const size_t, const size_t,
                                     const Buffer<double>&, const size_t, const size_t,
                                     const double,
                                     Buffer<double>&, const size_t, const size_t,
                                     const Queue&, Event*);
template void PUBLIC_API had<float2>(const size_t,
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
template void PUBLIC_API had<half>(const size_t,
                                   const half,
                                   const Buffer<half>&, const size_t, const size_t,
                                   const Buffer<half>&, const size_t, const size_t,
                                   const half,
                                   Buffer<half>&, const size_t, const size_t,
                                   const Queue&, Event*);

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
template <typename T>
void omatcopy(const Layout layout, const Transpose a_transpose,
              const size_t m, const size_t n,
              const T alpha,
              const Buffer<T>& a_buffer, const size_t a_offset, const size_t a_ld,
              Buffer<T>& b_buffer, const size_t b_offset, const size_t b_ld,
              const Queue& queue, Event* event) {
    auto routine = Xomatcopy<T>(queue, event);
    routine.DoOmatcopy(layout, a_transpose,
                       m, n,
                       alpha,
                       a_buffer, a_offset, a_ld,
                       b_buffer, b_offset, b_ld);
}

template void PUBLIC_API omatcopy<float>(const Layout, const Transpose,
                                         const size_t, const size_t,
                                         const float,
                                         const Buffer<float>&, const size_t, const size_t,
                                         Buffer<float>&, const size_t, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API omatcopy<double>(const Layout, const Transpose,
                                          const size_t, const size_t,
                                          const double,
                                          const Buffer<double>&, const size_t, const size_t,
                                          Buffer<double>&, const size_t, const size_t,
                                          const Queue&, Event*);
template void PUBLIC_API omatcopy<float2>(const Layout, const Transpose,
                                          const size_t, const size_t,
                                          const float2,
                                          const Buffer<float2>&, const size_t, const size_t,
                                          Buffer<float2>&, const size_t, const size_t,
                                          const Queue&, Event*);
template void PUBLIC_API omatcopy<double2>(const Layout, const Transpose,
                                           const size_t, const size_t,
                                           const double2,
                                           const Buffer<double2>&, const size_t, const size_t,
                                           Buffer<double2>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API omatcopy<half>(const Layout, const Transpose,
                                        const size_t, const size_t,
                                        const half,
                                        const Buffer<half>&, const size_t, const size_t,
                                        Buffer<half>&, const size_t, const size_t,
                                        const Queue&, Event*);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
template <typename T>
void im2col(const KernelMode kernel_mode,
            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
            const Buffer<T>& im_buffer, const size_t im_offset,
            Buffer<T>& col_buffer, const size_t col_offset,
            const Queue& queue, Event* event) {
    auto routine = Xim2col<T>(queue, event);
    routine.DoIm2col(kernel_mode,
                     channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                     im_buffer, im_offset,
                     col_buffer, col_offset);
}

template void PUBLIC_API im2col<float>(const KernelMode,
                                       const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                       const Buffer<float>&, const size_t,
                                       Buffer<float>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API im2col<double>(const KernelMode,
                                        const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                        const Buffer<double>&, const size_t,
                                        Buffer<double>&, const size_t,
                                        const Queue&, Event*);
template void PUBLIC_API im2col<float2>(const KernelMode,
                                        const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                        const Buffer<float2>&, const size_t,
                                        Buffer<float2>&, const size_t,
                                        const Queue&, Event*);
template void PUBLIC_API im2col<double2>(const KernelMode,
                                         const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                         const Buffer<double2>&, const size_t,
                                         Buffer<double2>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API im2col<half>(const KernelMode,
                                      const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t, const size_t,
                                      const Buffer<half>&, const size_t,
                                      Buffer<half>&, const size_t,
                                      const Queue&, Event*);

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

template void PUBLIC_API col2im<float>(const KernelMode,
                                       const size_t, const size_t, const size_t,
                                       const size_t, const size_t,
                                       const size_t, const size_t,
                                       const size_t, const size_t,
                                       const size_t, const size_t,
                                       const Buffer<float>&, const size_t,
                                       Buffer<float>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API col2im<double>(const KernelMode,
                                        const size_t, const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const Buffer<double>&, const size_t,
                                        Buffer<double>&, const size_t,
                                        const Queue&, Event*);
template void PUBLIC_API col2im<float2>(const KernelMode,
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
template void PUBLIC_API col2im<half>(const KernelMode,
                                      const size_t, const size_t, const size_t,
                                      const size_t, const size_t,
                                      const size_t, const size_t,
                                      const size_t, const size_t,
                                      const size_t, const size_t,
                                      const Buffer<half>&, const size_t,
                                      Buffer<half>&, const size_t,
                                      const Queue&, Event*);

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
              const Queue& queue, Event* event) {
    auto routine = Xconvgemm<T>(queue, event);
    routine.DoConvgemm(kernel_mode,
                       channels, height, width,
                       kernel_h, kernel_w,
                       pad_h, pad_w,
                       stride_h, stride_w,
                       dilation_h, dilation_w,
                       num_kernels, batch_count,
                       im_buffer, im_offset,
                       kernel_buffer, kernel_offset,
                       result_buffer, result_offset);
}

template void PUBLIC_API convgemm<float>(const KernelMode,
                                         const size_t, const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const size_t, const size_t,
                                         const Buffer<float>&, const size_t,
                                         const Buffer<float>&, const size_t,
                                         Buffer<float>&, const size_t,
                                         const Queue&, Event*);
template void PUBLIC_API convgemm<double>(const KernelMode,
                                          const size_t, const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const size_t, const size_t,
                                          const Buffer<double>&, const size_t,
                                          const Buffer<double>&, const size_t,
                                          Buffer<double>&, const size_t,
                                          const Queue&, Event*);
template void PUBLIC_API convgemm<half>(const KernelMode,
                                        const size_t, const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const Buffer<half>&, const size_t,
                                        const Buffer<half>&, const size_t,
                                        Buffer<half>&, const size_t,
                                        const Queue&, Event*);

// Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED
template <typename T>
void axpyBatched(const size_t n,
                 const T *alphas,
                 const Buffer<T>& x_buffer, const size_t *x_offsets, const size_t x_inc,
                 Buffer<T>& y_buffer, const size_t *y_offsets, const size_t y_inc,
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
    routine.DoAxpyBatched(n,
                          alphas_cpp,
                          x_buffer, x_offsets_cpp, x_inc,
                          y_buffer, y_offsets_cpp, y_inc,
                          batch_count);
}

template void PUBLIC_API axpyBatched<float>(const size_t,
                                            const float*,
                                            const Buffer<float>&, const size_t*, const size_t,
                                            Buffer<float>&, const size_t*, const size_t,
                                            const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API axpyBatched<double>(const size_t,
                                             const double*,
                                             const Buffer<double>&, const size_t*, const size_t,
                                             Buffer<double>&, const size_t*, const size_t,
                                             const size_t,
                                             const Queue&, Event*);
template void PUBLIC_API axpyBatched<float2>(const size_t,
                                             const float2*,
                                             const Buffer<float2>&, const size_t*, const size_t,
                                             Buffer<float2>&, const size_t*, const size_t,
                                             const size_t,
                                             const Queue&, Event*);
template void PUBLIC_API axpyBatched<double2>(const size_t,
                                              const double2*,
                                              const Buffer<double2>&, const size_t*, const size_t,
                                              Buffer<double2>&, const size_t*, const size_t,
                                              const size_t,
                                              const Queue&, Event*);
template void PUBLIC_API axpyBatched<half>(const size_t,
                                           const half*,
                                           const Buffer<half>&, const size_t*, const size_t,
                                           Buffer<half>&, const size_t*, const size_t,
                                           const size_t,
                                           const Queue&, Event*);

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
                 const Queue& queue, Event* event) {
    auto routine = XgemmBatched<T>(queue, event);
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
    routine.DoGemmBatched(layout, a_transpose, b_transpose,
                          m, n, k,
                          alphas_cpp,
                          a_buffer, a_offsets_cpp, a_ld,
                          b_buffer, b_offsets_cpp, b_ld,
                          betas_cpp,
                          c_buffer, c_offsets_cpp, c_ld,
                          batch_count);
}

template void PUBLIC_API gemmBatched<float>(const Layout, const Transpose, const Transpose,
                                            const size_t, const size_t, const size_t,
                                            const float*,
                                            const Buffer<float>&, const size_t*, const size_t,
                                            const Buffer<float>&, const size_t*, const size_t,
                                            const float*,
                                            Buffer<float>&, const size_t*, const size_t,
                                            const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API gemmBatched<double>(const Layout, const Transpose, const Transpose,
                                             const size_t, const size_t, const size_t,
                                             const double*,
                                             const Buffer<double>&, const size_t*, const size_t,
                                             const Buffer<double>&, const size_t*, const size_t,
                                             const double*,
                                             Buffer<double>&, const size_t*, const size_t,
                                             const size_t,
                                             const Queue&, Event*);
template void PUBLIC_API gemmBatched<float2>(const Layout, const Transpose, const Transpose,
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
template void PUBLIC_API gemmBatched<half>(const Layout, const Transpose, const Transpose,
                                           const size_t, const size_t, const size_t,
                                           const half*,
                                           const Buffer<half>&, const size_t*, const size_t,
                                           const Buffer<half>&, const size_t*, const size_t,
                                           const half*,
                                           Buffer<half>&, const size_t*, const size_t,
                                           const size_t,
                                           const Queue&, Event*);

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
    auto routine = XgemmStridedBatched<T>(queue, event);
    routine.DoGemmStridedBatched(layout, a_transpose, b_transpose,
                                 m, n, k,
                                 alpha,
                                 a_buffer, a_offset, a_ld, a_stride,
                                 b_buffer, b_offset, b_ld, b_stride,
                                 beta,
                                 c_buffer, c_offset, c_ld, c_stride,
                                 batch_count);
}

template void PUBLIC_API gemmStridedBatched<float>(const Layout, const Transpose, const Transpose,
                                                   const size_t, const size_t, const size_t,
                                                   const float,
                                                   const Buffer<float>&, const size_t, const size_t, const size_t,
                                                   const Buffer<float>&, const size_t, const size_t, const size_t,
                                                   const float,
                                                   Buffer<float>&, const size_t, const size_t, const size_t,
                                                   const size_t,
                                                   const Queue&, Event*);
template void PUBLIC_API gemmStridedBatched<double>(const Layout, const Transpose, const Transpose,
                                                    const size_t, const size_t, const size_t,
                                                    const double,
                                                    const Buffer<double>&, const size_t, const size_t, const size_t,
                                                    const Buffer<double>&, const size_t, const size_t, const size_t,
                                                    const double,
                                                    Buffer<double>&, const size_t, const size_t, const size_t,
                                                    const size_t,
                                                    const Queue&, Event*);
template void PUBLIC_API gemmStridedBatched<float2>(const Layout, const Transpose, const Transpose,
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
template void PUBLIC_API gemmStridedBatched<half>(const Layout, const Transpose, const Transpose,
                                                  const size_t, const size_t, const size_t,
                                                  const half,
                                                  const Buffer<half>&, const size_t, const size_t, const size_t,
                                                  const Buffer<half>&, const size_t, const size_t, const size_t,
                                                  const half,
                                                  Buffer<half>&, const size_t, const size_t, const size_t,
                                                  const size_t,
                                                  const Queue&, Event*);

// =================================================================================================

// Retrieves the required size of the temporary buffer for the GEMM kernel (optional)
template <typename T>
void gemmTempBufferSize(const Layout layout, const Transpose a_transpose, const Transpose b_transpose,
                        const size_t m, const size_t n, const size_t k,
                        const size_t a_offset, const size_t a_ld,
                        const size_t b_offset, const size_t b_ld,
                        const size_t c_offset, const size_t c_ld,
                        const Queue& queue, size_t& temp_buffer_size) {
    // Retrieves the tuning database
    const auto device = queue.context().device(); // FIXME
    const auto kernel_names = std::vector<std::string>{"Xgemm", "GemmRoutine"};
    Databases db(kernel_names);
    Routine::InitDatabase(device, kernel_names, PrecisionValue<T>(), {}, db);

    // Computes the buffer size
    if (Xgemm<T>::UseDirectKernel(m, n, k, db["XGEMM_MIN_INDIRECT_SIZE"])) {
      temp_buffer_size = 0;
    }
    else {
      temp_buffer_size = Xgemm<T>::GetTempSize(layout, a_transpose, b_transpose, m, n, k,
                                               a_offset, a_ld, b_offset, b_ld, c_offset, c_ld,
                                               db["MWG"], db["NWG"], db["KWG"] * db["KREG"],
                                               db["GEMMK"]);
    }
    temp_buffer_size *= sizeof(T); // translate from num-elements to bytes
}

template void PUBLIC_API gemmTempBufferSize<float>(const Layout, const Transpose, const Transpose,
                                                   const size_t, const size_t, const size_t,
                                                   const size_t, const size_t, const size_t, const size_t,
                                                   const size_t, const size_t, const Queue&, size_t&);
template void PUBLIC_API gemmTempBufferSize<double>(const Layout, const Transpose, const Transpose,
                                                    const size_t, const size_t, const size_t,
                                                    const size_t, const size_t, const size_t, const size_t,
                                                    const size_t, const size_t, const Queue&, size_t&);
template void PUBLIC_API gemmTempBufferSize<float2>(const Layout, const Transpose, const Transpose,
                                                    const size_t, const size_t, const size_t,
                                                    const size_t, const size_t, const size_t, const size_t,
                                                    const size_t, const size_t, const Queue&, size_t&);
template void PUBLIC_API gemmTempBufferSize<double2>(const Layout, const Transpose, const Transpose,
                                                     const size_t, const size_t, const size_t,
                                                     const size_t, const size_t, const size_t, const size_t,
                                                     const size_t, const size_t, const Queue&, size_t&);
template void PUBLIC_API gemmTempBufferSize<half>(const Layout, const Transpose, const Transpose,
                                                  const size_t, const size_t, const size_t,
                                                  const size_t, const size_t, const size_t, const size_t,
                                                  const size_t, const size_t, const Queue&, size_t&);

// =================================================================================================
} // namespace gpgpu::blas
