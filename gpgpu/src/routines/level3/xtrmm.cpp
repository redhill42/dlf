
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xtrmm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xtrmm.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xtrmm<T>::Xtrmm(const Queue &queue, Event* event, const std::string &name):
    Xgemm<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xtrmm<T>::DoTrmm(const Layout layout, const Side side, const Triangle triangle,
                      const Transpose a_transpose, const Diagonal diagonal,
                      const size_t m, const size_t n,
                      const T alpha,
                      const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                      Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld) {

  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0)) { throw BLASError(StatusCode::kInvalidDimension); }

  // Computes the k dimension. This is based on whether or not matrix is A (on the left)
  // or B (on the right) in the Xgemm routine.
  auto k = (side == Side::Left) ? m : n;

  // Checks for validity of the triangular A matrix
  TestMatrixA(k, k, a_buffer, a_offset, a_ld);

  // Checks for validity of the input/output B matrix
  const auto b_one = (layout == Layout::RowMajor) ? n : m;
  const auto b_two = (layout == Layout::RowMajor) ? m : n;
  TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);

  // Creates a copy of B to avoid overwriting input in GEMM while computing output
  const auto b_size = b_ld * (b_two - 1) + b_one;
  auto b_buffer_copy = context_.template getTemporaryBuffer<T>(b_size);
  b_buffer.copyTo(queue_, b_buffer_copy, b_size, b_offset, b_buffer_copy.offset());

  // Determines which kernel to run based on the layout (the Xgemm kernel assumes column-major as
  // default) and on whether we are dealing with an upper or lower triangle of the triangular matrix
  bool is_upper = ((triangle == Triangle::Upper && layout != Layout::RowMajor) ||
                   (triangle == Triangle::Lower && layout == Layout::RowMajor));
  auto kernel_name = (is_upper) ? "TriaUpperToSquared" : "TriaLowerToSquared";

  // Determines whether or not the triangular matrix is unit-diagonal
  auto unit_diagonal = (diagonal == Diagonal::Unit) ? true : false;

  // Temporary buffer for a copy of the triangular matrix
  auto temp_triangular = context_.template getTemporaryBuffer<T>(k*k);

  // Creates a general matrix from the triangular matrix to be able to run the regular Xgemm
  // routine afterwards
  auto kernel = program_.getKernel(kernel_name);

  // Sets the arguments for the triangular-to-squared kernel
  kernel.setArgument(0, static_cast<int>(k));
  kernel.setArgument(1, static_cast<int>(a_ld));
  kernel.setArgument(2, static_cast<int>(a_offset));
  kernel.setArgument(3, a_buffer);
  kernel.setArgument(4, static_cast<int>(k));
  kernel.setArgument(5, static_cast<int>(k));
  kernel.setArgument(6, static_cast<int>(temp_triangular.offset()));
  kernel.setArgument(7, temp_triangular);
  kernel.setArgument(8, static_cast<int>(unit_diagonal));

  // Uses the common padding kernel's thread configuration. This is allowed, since the
  // triangular-to-squared kernel uses the same parameters.
  auto global = std::vector<size_t>{Ceil(CeilDiv(k, db_["PAD_WPTX"]), db_["PAD_DIMX"]),
                                    Ceil(CeilDiv(k, db_["PAD_WPTY"]), db_["PAD_DIMY"])};
  auto local = std::vector<size_t>{db_["PAD_DIMX"], db_["PAD_DIMY"]};
  RunKernel(kernel, queue_, device_, global, local, nullptr);

  // Runs the regular Xgemm code with either "B := alpha*A*B" or ...
  if (side == Side::Left) {
    DoGemm(layout, a_transpose, Transpose::NoTrans,
           m, n, k,
           alpha,
           temp_triangular, temp_triangular.offset(), k,
           b_buffer_copy, b_buffer_copy.offset(), b_ld,
           ConstantZero<T>(),
           b_buffer, b_offset, b_ld);
  }

  // ... with "B := alpha*B*A". Note that A and B are now reversed.
  else {
    try {
      DoGemm(layout, Transpose::NoTrans, a_transpose,
             m, n, k,
             alpha,
             b_buffer_copy, b_buffer_copy.offset(), b_ld,
             temp_triangular, temp_triangular.offset(), k,
             ConstantZero<T>(),
             b_buffer, b_offset, b_ld);
    } catch (BLASError &e) {
      // A and B are now reversed, so also reverse the error codes returned from the Xgemm routine
      switch(e.status()) {
        case StatusCode::kInvalidMatrixA:      throw BLASError(StatusCode::kInvalidMatrixB, e.details());
        case StatusCode::kInvalidMatrixB:      throw BLASError(StatusCode::kInvalidMatrixA, e.details());
        case StatusCode::kInvalidLeadDimA:     throw BLASError(StatusCode::kInvalidLeadDimB, e.details());
        case StatusCode::kInvalidLeadDimB:     throw BLASError(StatusCode::kInvalidLeadDimA, e.details());
        case StatusCode::kInsufficientMemoryA: throw BLASError(StatusCode::kInsufficientMemoryB, e.details());
        case StatusCode::kInsufficientMemoryB: throw BLASError(StatusCode::kInsufficientMemoryA, e.details());
        default:                               throw;
      }
    }
  }
}

// =================================================================================================

// Compiles the templated class
template class Xtrmm<half>;
template class Xtrmm<float>;
template class Xtrmm<double>;
template class Xtrmm<float2>;
template class Xtrmm<double2>;
template class Xtrmm<int32_t>;
template class Xtrmm<int64_t>;

// =================================================================================================
}} // namespace gpgpu::blas
