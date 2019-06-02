
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xomatcopy routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XOMATCOPY_H_
#define GPGPU_BLAS_ROUTINES_XOMATCOPY_H_

#include "routine.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xomatcopy: public Routine {
 public:

  // Constructor
  Xomatcopy(const Queue &queue, Event* event, const std::string &name = "OMATCOPY");

  // Templated-precision implementation of the routine
  void DoOmatcopy(const Layout layout, const Transpose a_transpose,
                  const size_t m, const size_t n, const T alpha,
                  const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                  Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld);
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XOMATCOPY_H_
#endif
