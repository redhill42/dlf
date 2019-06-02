
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xcol2im class (see the header for information about the class).
//
// =================================================================================================

#include "routines/levelx/xcol2im.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xcol2im<T>::Xcol2im(const Queue &queue, Event* event, const std::string &name):
    Routine(queue, event, name, {"Copy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/levelx/col2im.cl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xcol2im<T>::DoCol2im(const KernelMode kernel_mode,
                          const size_t channels, const size_t height, const size_t width,
                          const size_t kernel_h, const size_t kernel_w, const size_t pad_h,
                          const size_t pad_w, const size_t stride_h, const size_t stride_w,
                          const size_t dilation_h, const size_t dilation_w,
                          const Buffer<T> &col_buffer, const size_t col_offset,
                          Buffer<T> &im_buffer, const size_t im_offset) {

  // Flip the output along kernel_h and kernel_w, or not.
  const auto kernel_name = (kernel_mode == KernelMode::Convolution) ? "Xcol2imKernelFlip" : "Xcol2imKernelNormal";

  // Makes sure all dimensions are larger than zero
  if ((channels == 0) || (height == 0) || (width == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

  // Sets the output height and width
  const auto size_h = height + 2 * pad_h;
  const auto padding_h = dilation_h * (kernel_h - 1) + 1;
  const auto col_h = (size_h >= padding_h) ? (size_h - padding_h) / stride_h + 1 : 1;
  const auto size_w = width + 2 * pad_w;
  const auto padding_w = dilation_w * (kernel_w - 1) + 1;
  const auto col_w = (size_w >= padding_w) ? (size_w - padding_w) / stride_w + 1 : 1;

  int stride_bez_h = 0;
  int stride_bez_w = 0;
  int dilation_bez_h = 0;
  int dilation_bez_w = 0;
  int gcd_h = 0;
  int gcd_w = 0;
  EuclidGCD(static_cast<int>(stride_h), static_cast<int>(dilation_h), stride_bez_h, dilation_bez_h, gcd_h);
  EuclidGCD(static_cast<int>(stride_w), static_cast<int>(dilation_w), stride_bez_w, dilation_bez_w, gcd_w);

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
  kernel.setArgument(13, stride_bez_h);
  kernel.setArgument(14, stride_bez_w);
  kernel.setArgument(15, dilation_bez_h);
  kernel.setArgument(16, dilation_bez_w);
  kernel.setArgument(17, gcd_h);
  kernel.setArgument(18, gcd_w);
  kernel.setArgument(19, col_buffer);
  kernel.setArgument(20, static_cast<int>(col_offset));
  kernel.setArgument(21, im_buffer);
  kernel.setArgument(22, static_cast<int>(im_offset));

  // Launches the kernel
  const auto w_ceiled = Ceil((width - 1) / gcd_w + 1, db_["COPY_DIMX"]);
  const auto h_ceiled = Ceil((height - 1) / gcd_h + 1, db_["COPY_DIMY"]);
  const auto global = std::vector<size_t>{w_ceiled, h_ceiled * channels};
  const auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
  RunKernel(kernel, queue_, device_, global, local, event_);
}

// =================================================================================================

// Compiles the templated class
template class Xcol2im<half>;
template class Xcol2im<float>;
template class Xcol2im<double>;
template class Xcol2im<float2>;
template class Xcol2im<double2>;

// =================================================================================================
}} // namespace gpgpu::blas
