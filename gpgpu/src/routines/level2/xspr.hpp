
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xspr routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XSPR_H_
#define GPGPU_BLAS_ROUTINES_XSPR_H_

#include "routines/level2/xher.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xspr: public Xher<T,T> {
 public:

  // Uses the regular Xher routine
  using Xher<T,T>::DoHer;

  // Constructor
  Xspr(const Queue &queue, Event* event, const std::string &name = "SPR");

  // Templated-precision implementation of the routine
  void DoSpr(const Layout layout, const Triangle triangle,
             const size_t n,
             const T alpha,
             const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
             Buffer<T> &ap_buffer, const size_t ap_offset);
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XSPR_H_
#endif
