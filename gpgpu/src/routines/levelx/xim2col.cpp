
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

namespace gpgpu::blas {
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
                          const size_t channels, const size_t height, const size_t width,
                          const size_t kernel_h, const size_t kernel_w, const size_t pad_h,
                          const size_t pad_w, const size_t stride_h, const size_t stride_w,
                          const size_t dilation_h, const size_t dilation_w,
                          const Buffer<T> &im_buffer, const size_t im_offset,
                          Buffer<T> &col_buffer, const size_t col_offset) {

  // Flip the output along kernel_h and kernel_w, or not.
  const auto kernel_name = (kernel_mode == KernelMode::Convolution) ? "Xim2colKernelFlip" : "Xim2colKernelNormal";

  // Makes sure all dimensions are larger than zero
  if ((channels == 0) || (height == 0) || (width == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

  // Sets the height and width of the 'col' result
  const auto size_h = height + 2 * pad_h;
  const auto padding_h = dilation_h * (kernel_h - 1) + 1;
  const auto col_h = (size_h >= padding_h) ? (size_h - padding_h) / stride_h + 1 : 1;
  const auto size_w = width + 2 * pad_w;
  const auto padding_w = dilation_w * (kernel_w - 1) + 1;
  const auto col_w = (size_w >= padding_w) ? (size_w - padding_w) / stride_w + 1 : 1;

  // Retrieves the kernel from the compiled binary
  auto kernel = program_.getKernel(kernel_name);

  // Sets the kernel arguments
  kernel.setArgument(0, static_cast<int>(height));
  kernel.setArgument(1, static_cast<int>(width));
  kernel.setArgument(2, static_cast<int>(channels));
  kernel.setArgument(3, static_cast<int>(col_h));
  kernel.setArgument(4, static_cast<int>(col_w));
  kernel.setArgument(5, static_cast<int>(kernel_h));
  kernel.setArgument(6, static_cast<int>(kernel_w));
  kernel.setArgument(7, static_cast<int>(pad_h));
  kernel.setArgument(8, static_cast<int>(pad_w));
  kernel.setArgument(9, static_cast<int>(stride_h));
  kernel.setArgument(10, static_cast<int>(stride_w));
  kernel.setArgument(11, static_cast<int>(dilation_h));
  kernel.setArgument(12, static_cast<int>(dilation_w));
  kernel.setArgument(13, im_buffer);
  kernel.setArgument(14, static_cast<int>(im_offset));
  kernel.setArgument(15, col_buffer);
  kernel.setArgument(16, static_cast<int>(col_offset));

  // Launches the kernel
  const auto w_ceiled = Ceil(col_w, db_["COPY_DIMX"]);
  const auto h_ceiled = Ceil(col_h, db_["COPY_DIMY"]);
  const auto global = std::vector<size_t>{w_ceiled, h_ceiled * channels};
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
} // namespace gpgpu::blas
