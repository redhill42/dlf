
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xger routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XGER_H_
#define GPGPU_BLAS_ROUTINES_XGER_H_

#include "routine.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xger: public Routine {
 public:

  // Constructor
  Xger(const Queue &queue, Event* event, const std::string &name = "GER");

  // Templated-precision implementation of the routine
  void DoGer(const Layout layout,
             const size_t m, const size_t n,
             const T alpha,
             const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
             const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
             Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld);
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XGER_H_
#endif
