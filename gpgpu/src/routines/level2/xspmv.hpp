
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xspmv routine. It is based on the generalized mat-vec multiplication
// routine (Xgemv). The Xspmv class inherits from the templated class Xgemv, allowing it to call the
// "MatVec" function directly.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XSPMV_H_
#define GPGPU_BLAS_ROUTINES_XSPMV_H_

#include "routines/level2/xgemv.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xspmv: public Xgemv<T> {
 public:

  // Uses the generic matrix-vector routine
  using Xgemv<T>::MatVec;

  // Constructor
  Xspmv(const Queue &queue, Event* event, const std::string &name = "SPMV");

  // Templated-precision implementation of the routine
  void DoSpmv(const Layout layout, const Triangle triangle,
              const size_t n,
              const T alpha,
              const Buffer<T> &ap_buffer, const size_t ap_offset,
              const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
              const T beta,
              Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc);
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XSPMV_H_
#endif
