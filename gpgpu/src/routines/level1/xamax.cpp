
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xamax class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xamax.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xamax<T>::Xamax(const Queue &queue, Event* event, const std::string &name):
    Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/xamax.cl"
    }) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xamax<T>::DoAmax(const size_t n,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      Buffer<unsigned int> &imax_buffer, const size_t imax_offset) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Tests the vectors for validity
  TestVectorX(n, x_buffer, x_offset, x_inc);
  TestVectorIndex(1, imax_buffer, imax_offset);

  // Retrieves the Xamax kernels from the compiled binary
  auto kernel1 = program_.getKernel("Xamax");
  auto kernel2 = program_.getKernel("XamaxEpilogue");

  // Creates the buffer for intermediate values
  auto temp_size = 2*db_["WGS2"];
  auto temp_buffer1 = context_.createBuffer<T>(temp_size);
  auto temp_buffer2 = context_.createBuffer<unsigned int>(temp_size);

  // Sets the kernel arguments
  kernel1.setArgument(0, static_cast<int>(n));
  kernel1.setArgument(1, x_buffer);
  kernel1.setArgument(2, static_cast<int>(x_offset));
  kernel1.setArgument(3, static_cast<int>(x_inc));
  kernel1.setArgument(4, temp_buffer1);
  kernel1.setArgument(5, temp_buffer2);

  // Launches the main kernel
  auto global1 = std::vector<size_t>{db_["WGS1"]*temp_size};
  auto local1 = std::vector<size_t>{db_["WGS1"]};
  RunKernel(kernel1, queue_, device_, global1, local1, nullptr);

  // Sets the arguments for the epilogue kernel
  kernel2.setArgument(0, temp_buffer1);
  kernel2.setArgument(1, temp_buffer2);
  kernel2.setArgument(2, imax_buffer);
  kernel2.setArgument(3, static_cast<int>(imax_offset));

  // Launches the epilogue kernel
  auto global2 = std::vector<size_t>{db_["WGS2"]};
  auto local2 = std::vector<size_t>{db_["WGS2"]};
  RunKernel(kernel2, queue_, device_, global2, local2, event_);
}

// =================================================================================================

// Compiles the templated class
template class Xamax<half>;
template class Xamax<float>;
template class Xamax<double>;
template class Xamax<float2>;
template class Xamax<double2>;
template class Xamax<int32_t>;
template class Xamax<int64_t>;

// =================================================================================================
}} // namespace gpgpu::blas
