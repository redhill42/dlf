
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xim2col routine. The precision is implemented using a template argument.
// Uses the tuning parameters from the regular copy kernel.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XIM2COL_H_
#define GPGPU_BLAS_ROUTINES_XIM2COL_H_

#include "routine.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T>
class Xim2col: public Routine {
 public:

  // Constructor
  Xim2col(const Queue &queue, Event* event, const std::string &name = "IM2COL");

  // Templated-precision implementation of the routine
  void DoIm2col(const KernelMode kernel_mode,
                const size_t channels, const size_t height, const size_t width,
                const size_t output_h, const size_t output_w,
                const size_t kernel_h, const size_t kernel_w,
                const size_t pad_h, const size_t pad_w,
                const size_t stride_h, const size_t stride_w,
                const size_t dilation_h, const size_t dilation_w,
                const Buffer<T> &im_buffer, const size_t im_offset,
                Buffer<T> &col_buffer, const size_t col_offset);
};

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINES_XIM2COL_H_
#endif
