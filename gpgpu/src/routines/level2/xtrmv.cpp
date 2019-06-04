
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrmv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xtrmv.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xtrmv<T>::Xtrmv(const Queue &queue, Event* event, const std::string &name):
    Xgemv<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xtrmv<T>::DoTrmv(const Layout layout, const Triangle triangle,
                      const Transpose a_transpose, const Diagonal diagonal,
                      const size_t n,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc) {

  // Creates a copy of X: a temporary scratch buffer
  auto scratch_buffer = context_.template createBuffer<T>(n*x_inc + x_offset);
  x_buffer.copyTo(queue_, scratch_buffer, n*x_inc + x_offset);

  // The data is either in the upper or lower triangle
  size_t is_upper = ((triangle == Triangle::Upper && layout != Layout::RowMajor) ||
                     (triangle == Triangle::Lower && layout == Layout::RowMajor));

  // Adds '2' to the parameter if the diagonal is unit
  auto parameter = (diagonal == Diagonal::Unit) ? is_upper + 2 : is_upper;

  // Runs the generic matrix-vector multiplication, disabling the use of fast vectorized kernels.
  // The specific triangular matrix-accesses are implemented in the kernel guarded by the
  // ROUTINE_TRMV define.
  auto fast_kernels = false;
  try {
    MatVec(layout, a_transpose,
           n, n, ConstantOne<T>(),
           a_buffer, a_offset, a_ld,
           scratch_buffer, x_offset, x_inc, ConstantZero<T>(),
           x_buffer, x_offset, x_inc,
           fast_kernels, fast_kernels,
           parameter, false, 0, 0);
  } catch (BLASError &e) {
    // Returns the proper error code (renames vector Y to X)
    switch (e.status()) {
      case StatusCode::kInvalidVectorY:      throw BLASError(StatusCode::kInvalidVectorX, e.details());
      case StatusCode::kInvalidIncrementY:   throw BLASError(StatusCode::kInvalidIncrementX, e.details());
      case StatusCode::kInsufficientMemoryY: throw BLASError(StatusCode::kInsufficientMemoryX, e.details());
      default:                               throw;
    }
  }
}

// =================================================================================================

// Compiles the templated class
template class Xtrmv<half>;
template class Xtrmv<float>;
template class Xtrmv<double>;
template class Xtrmv<float2>;
template class Xtrmv<double2>;

// =================================================================================================
}} // namespace gpgpu::blas
