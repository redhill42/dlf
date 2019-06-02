
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xmax routine. The precision is implemented using a template argument.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XMAX_H_
#define GPGPU_BLAS_ROUTINES_XMAX_H_

#include "routine.hpp"
#include "routines/level1/xamax.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xmax: public Xamax<T> {
 public:

  // Members and methods from the base class
  using Xamax<T>::DoAmax;

  // Constructor
  Xmax(const Queue &queue, Event* event, const std::string &name = "MAX"):
    Xamax<T>(queue, event, name) {
  }

  // Forwards to the regular absolute version. The implementation difference is realised in the
  // kernel through a pre-processor macro based on the name of the routine.
  void DoMax(const size_t n,
             const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
             Buffer<unsigned int> &imax_buffer, const size_t imax_offset) {
    DoAmax(n, x_buffer, x_offset, x_inc, imax_buffer, imax_offset);
  }
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XMAX_H_
#endif
