
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xamax routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XAMAX_H_
#define GPGPU_BLAS_ROUTINES_XAMAX_H_

#include "routine.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xamax: public Routine {
 public:

  // Constructor
  Xamax(const Queue &queue, Event* event, const std::string &name = "AMAX");

  // Templated-precision implementation of the routine
  void DoAmax(const size_t n,
              const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
              Buffer<unsigned int> &imax_buffer, const size_t imax_offset);
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XAMAX_H_
#endif
