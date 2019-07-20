
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xim2col class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xim2col.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xim2col<T>::Xim2col(const Queue &queue, Event* event, const std::string &name):
    Routine(queue, event, name, {"Copy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/levelx/im2col.cl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xim2col<T>::DoIm2col(const KernelMode kernel_mode,
                          const size_t batches, const size_t channels,
                          const size_t height, const size_t width,
                          const size_t output_h, const size_t output_w,
                          const size_t kernel_h, const size_t kernel_w,
                          const size_t pad_h, const size_t pad_w,
                          const size_t stride_h, const size_t stride_w,
                          const size_t dilation_h, const size_t dilation_w,
                          const Buffer<T> &im_buffer, const size_t im_offset,
                          Buffer<T> &col_buffer, const size_t col_offset) {

  // Flip the output along kernel_h and kernel_w, or not.
  const auto kernel_name = (kernel_mode == KernelMode::Convolution) ? "Xim2colKernelFlip" : "Xim2colKernelNormal";

  // Makes sure all dimensions are larger than zero
  if ((channels == 0) || (height == 0) || (width == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

  // Retrieves the kernel from the compiled binary
  auto kernel = program_.getKernel(kernel_name);

  // Sets the kernel arguments
  kernel.setArguments(static_cast<int>(height),
                      static_cast<int>(width),
                      static_cast<int>(batches),
                      static_cast<int>(channels),
                      static_cast<int>(output_h),
                      static_cast<int>(output_w),
                      static_cast<int>(kernel_h),
                      static_cast<int>(kernel_w),
                      static_cast<int>(pad_h),
                      static_cast<int>(pad_w),
                      static_cast<int>(stride_h),
                      static_cast<int>(stride_w),
                      static_cast<int>(dilation_h),
                      static_cast<int>(dilation_w),
                      im_buffer, static_cast<int>(im_offset),
                      col_buffer, static_cast<int>(col_offset));

  // Launches the kernel
  const auto w_ceiled = Ceil(output_w, db_["COPY_DIMX"]);
  const auto h_ceiled = Ceil(output_h, db_["COPY_DIMY"]);
  const auto global = std::vector<size_t>{w_ceiled * batches, h_ceiled * channels};
  const auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
  RunKernel(kernel, queue_, device_, global, local, event_);
}

// =================================================================================================

// Compiles the templated class
template class Xim2col<half>;
template class Xim2col<float>;
template class Xim2col<double>;
template class Xim2col<float2>;
template class Xim2col<double2>;

// =================================================================================================
}} // namespace gpgpu::blas
