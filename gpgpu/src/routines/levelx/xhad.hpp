
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xhad routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XHAD_H_
#define GPGPU_BLAS_ROUTINES_XHAD_H_

#include "routine.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xhad: public Routine {
public:

  // Constructor
  Xhad(const Queue &queue, Event* event, const std::string &name = "HAD");

  // Templated-precision implementation of the routine
  void DoHad(const size_t n, const T alpha,
             const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
             const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc, const T beta,
             Buffer<T> &z_buffer, const size_t z_offset, const size_t z_inc);
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XHAD_H_
#endif
