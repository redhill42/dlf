
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhpr2 routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XHPR2_H_
#define GPGPU_BLAS_ROUTINES_XHPR2_H_

#include "routines/level2/xher2.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xhpr2: public Xher2<T> {
 public:

  // Uses the regular Xher2 routine
  using Xher2<T>::DoHer2;

  // Constructor
  Xhpr2(const Queue &queue, Event* event, const std::string &name = "HPR2");

  // Templated-precision implementation of the routine
  void DoHpr2(const Layout layout, const Triangle triangle,
              const size_t n,
              const T alpha,
              const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
              const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
              Buffer<T> &ap_buffer, const size_t ap_offset);
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XHPR2_H_
#endif
