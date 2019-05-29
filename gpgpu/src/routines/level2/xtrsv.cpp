
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrsv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xtrsv.hpp"

#include <string>
#include <vector>

namespace gpgpu::blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xtrsv<T>::Xtrsv(const Queue &queue, Event* event, const std::string &name):
    Xgemv<T>(queue, event, name) {
}

// =================================================================================================

template <typename T>
void Xtrsv<T>::Substitution(const Layout layout, const Triangle triangle,
                            const Transpose a_transpose, const Diagonal diagonal,
                            const size_t n,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_inc,
                            const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                            Event* event) {

  if (n > db_["TRSV_BLOCK_SIZE"]) { throw BLASError(StatusCode::kUnexpectedError); };

  // Translates CLBlast arguments to 0/1 integers for the OpenCL kernel
  const auto is_unit_diagonal = (diagonal == Diagonal::NonUnit) ? 0 : 1;
  const auto is_transposed = ((a_transpose == Transpose::NoTrans && layout == Layout::ColMajor) ||
                              (a_transpose != Transpose::NoTrans && layout != Layout::ColMajor)) ? 0 : 1;
  const auto do_conjugate = (a_transpose == Transpose::ConjTrans) ? 1 : 0;

  // The data is either in the upper or lower triangle
  const auto is_upper = ((triangle == Triangle::Upper && a_transpose == Transpose::NoTrans) ||
                         (triangle == Triangle::Lower && a_transpose != Transpose::NoTrans));

  // Retrieves the kernel from the compiled binary
  const auto kernel_name = (is_upper) ? "trsv_backward" : "trsv_forward";
  auto kernel = program_.getKernel(kernel_name);

  // Sets the kernel arguments
  kernel.setArgument(0, static_cast<int>(n));
  kernel.setArgument(1, a_buffer);
  kernel.setArgument(2, static_cast<int>(a_offset));
  kernel.setArgument(3, static_cast<int>(a_ld));
  kernel.setArgument(4, b_buffer);
  kernel.setArgument(5, static_cast<int>(b_offset));
  kernel.setArgument(6, static_cast<int>(b_inc));
  kernel.setArgument(7, x_buffer);
  kernel.setArgument(8, static_cast<int>(x_offset));
  kernel.setArgument(9, static_cast<int>(x_inc));
  kernel.setArgument(10, static_cast<int>(is_transposed));
  kernel.setArgument(11, static_cast<int>(is_unit_diagonal));
  kernel.setArgument(12, static_cast<int>(do_conjugate));

  // Launches the kernel
  const auto local = std::vector<size_t>{db_["TRSV_BLOCK_SIZE"]};
  const auto global = std::vector<size_t>{Ceil(n, db_["TRSV_BLOCK_SIZE"])};
  RunKernel(kernel, queue_, device_, global, local, event);
}

// =================================================================================================

// The main routine
template <typename T>
void Xtrsv<T>::DoTrsv(const Layout layout, const Triangle triangle,
                      const Transpose a_transpose, const Diagonal diagonal,
                      const size_t n,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      Buffer<T> &b_buffer, const size_t b_offset, const size_t b_inc) {

  // Makes sure all dimensions are larger than zero
  if (n == 0) { throw BLASError(StatusCode::kInvalidDimension); }

  // Some parts of this kernel are not tunable and thus require some minimal OpenCL properties
  if (device_.maxWorkGroupSize() < 16) { // minimum of total local work size of 16
    throw RuntimeErrorCode(StatusCode::kNotImplemented);
  }

  // Tests the matrix and vector
  TestMatrixA(n, n, a_buffer, a_offset, a_ld);
  TestVectorX(n, b_buffer, b_offset, b_inc);

  // Creates a copy of B to avoid overwriting input while computing output
  // TODO: Make x with 0 offset and unit increment by creating custom copy-to and copy-from kernels
  const auto x_offset = b_offset;
  const auto x_inc = b_inc;
  const auto x_size = n*x_inc + x_offset;
  auto x_buffer = context_.template createBuffer<T>(x_size);
  b_buffer.copyTo(queue_, x_buffer, x_size);

  // Fills the output buffer with zeros
  auto eventWaitList = std::vector<Event>();
  auto fill_vector_event = context_.createEvent();
  FillVector(queue_, device_, program_, &fill_vector_event, eventWaitList,
             n, x_inc, x_offset, x_buffer, ConstantZero<T>(), 16);
  fill_vector_event.waitForCompletion();

  // Derives properties based on the arguments
  const auto is_upper = ((triangle == Triangle::Upper && a_transpose == Transpose::NoTrans) ||
                         (triangle == Triangle::Lower && a_transpose != Transpose::NoTrans));
  const auto is_transposed = ((layout == Layout::ColMajor && a_transpose == Transpose::NoTrans) ||
                              (layout != Layout::ColMajor && a_transpose != Transpose::NoTrans));

  // Loops over the blocks
  auto col = n; // the initial column position
  for (auto i = size_t{0}; i < n; i += db_["TRSV_BLOCK_SIZE"]) {
    const auto block_size = std::min(db_["TRSV_BLOCK_SIZE"], n - i);

    // Sets the next column position
    col = (is_upper) ? col - block_size : i;

    // Sets the offsets for upper or lower triangular
    const auto extra_offset_a = (is_transposed) ?
                                (is_upper ? col + (col+block_size)*a_ld : col) :
                                (is_upper ? col+block_size + col*a_ld : col*a_ld);
    const auto extra_offset_x = (is_upper) ? (col+block_size)*x_inc : 0;
    const auto extra_offset_b = col*x_inc;

    // Runs the GEMV routine to compute x' = A * x
    if (i > 0) {
      const auto gemv_m = (a_transpose == Transpose::NoTrans) ? block_size : i;
      const auto gemv_n = (a_transpose == Transpose::NoTrans) ? i : block_size;
      auto gemv_event = context_.createEvent();
      auto gemv = Xgemv<T>(queue_, &gemv_event);
      gemv.DoGemv(layout, a_transpose, gemv_m, gemv_n, ConstantOne<T>(),
                  a_buffer, a_offset + extra_offset_a, a_ld,
                  x_buffer, x_offset + extra_offset_x, x_inc, ConstantOne<T>(),
                  x_buffer, x_offset + extra_offset_b, x_inc);
      gemv_event.waitForCompletion();
    }

    // Rurns the triangular substitution for the block size
    auto sub_event = context_.createEvent();
    Substitution(layout, triangle, a_transpose, diagonal, block_size,
                 a_buffer, a_offset + col + col*a_ld, a_ld,
                 b_buffer, b_offset + col*b_inc, b_inc,
                 x_buffer, x_offset + col*x_inc, x_inc, &sub_event);
    sub_event.waitForCompletion();
  }

  // Retrieves the results
  x_buffer.copyToAsync(queue_, b_buffer, x_size, event_);
}

// =================================================================================================

// Compiles the templated class
template class Xtrsv<half>;
template class Xtrsv<float>;
template class Xtrsv<double>;
template class Xtrsv<float2>;
template class Xtrsv<double2>;

// =================================================================================================
} // namespace gpgpu::blas
