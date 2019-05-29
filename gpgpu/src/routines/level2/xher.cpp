
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xher.hpp"

#include <string>

namespace gpgpu::blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T, typename U>
Xher<T,U>::Xher(const Queue &queue, Event* event, const std::string &name):
    Routine(queue, event, name, {"Xger"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level2/level2.cl"
    #include "../../kernels/level2/xher.cl"
    }) {
}

// =================================================================================================

// Specializations to compute alpha of type 'T'
template <> float2 Xher<float2,float>::GetAlpha(const float alpha) { return float2{alpha, 0.0f}; }
template <> double2 Xher<double2,double>::GetAlpha(const double alpha) { return double2{alpha, 0.0}; }
template <> float Xher<float,float>::GetAlpha(const float alpha) { return alpha; }
template <> double Xher<double,double>::GetAlpha(const double alpha) { return alpha; }
template <> half Xher<half,half>::GetAlpha(const half alpha) { return alpha; }

// =================================================================================================

// The main routine
template <typename T, typename U>
void Xher<T,U>::DoHer(const Layout layout, const Triangle triangle,
                      const size_t n,
                      const U alpha,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      const bool packed) {

  // Makes sure the dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // The data is either in the upper or lower triangle
  const auto is_upper = ((triangle == Triangle::Upper && layout != Layout::RowMajor) ||
                         (triangle == Triangle::Lower && layout == Layout::RowMajor));
  const auto is_rowmajor = (layout == Layout::RowMajor);

  // Tests the matrix and the vectors for validity
  if (packed) { TestMatrixAP(n, a_buffer, a_offset); }
  else { TestMatrixA(n, n, a_buffer, a_offset, a_ld); }
  TestVectorX(n, x_buffer, x_offset, x_inc);

  // If alpha is zero an update is not required
  if (alpha == U{0}) { return; }

  // Creates a matching version of alpha
  const auto matching_alpha = GetAlpha(alpha);

  // Retrieves the kernel from the compiled binary
  auto kernel = program_.getKernel("Xher");

  // Sets the kernel arguments
  kernel.setArgument(0, static_cast<int>(n));
  kernel.setArgument(1, GetRealArg(matching_alpha));
  kernel.setArgument(2, x_buffer);
  kernel.setArgument(3, static_cast<int>(x_offset));
  kernel.setArgument(4, static_cast<int>(x_inc));
  kernel.setArgument(5, a_buffer);
  kernel.setArgument(6, static_cast<int>(a_offset));
  kernel.setArgument(7, static_cast<int>(a_ld));
  kernel.setArgument(8, static_cast<int>(is_upper));
  kernel.setArgument(9, static_cast<int>(is_rowmajor));

  // Launches the kernel
  auto global_one = Ceil(CeilDiv(n, db_["WPT"]), db_["WGS1"]);
  auto global_two = Ceil(CeilDiv(n, db_["WPT"]), db_["WGS2"]);
  auto global = std::vector<size_t>{global_one, global_two};
  auto local = std::vector<size_t>{db_["WGS1"], db_["WGS2"]};
  RunKernel(kernel, queue_, device_, global, local, event_);
}

// =================================================================================================

// Compiles the templated class
template class Xher<half, half>;
template class Xher<float, float>;
template class Xher<double, double>;
template class Xher<float2, float>;
template class Xher<double2, double>;

// =================================================================================================
} // namespace gpgpu::blas
