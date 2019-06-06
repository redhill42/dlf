
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xher2k class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xher2k.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T, typename U>
Xher2k<T,U>::Xher2k(const Queue &queue, Event* event, const std::string &name):
    Xherk<T,U>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T, typename U>
void Xher2k<T,U>::DoHer2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                          const size_t n, const size_t k,
                          const T alpha,
                          const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                          const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                          const U beta,
                          Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Runs the first matrix multiplication
  auto complex_beta = T{beta, static_cast<U>(0.0)};
  const auto negated_ab_transpose = (ab_transpose != Transpose::NoTrans) ? Transpose::NoTrans : Transpose::Trans;
  HerkAB(layout, triangle, ab_transpose, negated_ab_transpose, n, k, alpha,
         a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, complex_beta, c_buffer, c_offset, c_ld,
         nullptr, false);

  // Swaps the arguments for matrices A and B, sets 'beta' to 1, and conjugate alpha
  auto conjugate_alpha = T{alpha.real(), -alpha.imag()};
  auto complex_one = T{static_cast<U>(1.0), static_cast<U>(0.0)};
  HerkAB(layout, triangle, ab_transpose, negated_ab_transpose, n, k, conjugate_alpha,
         b_buffer, b_offset, b_ld, a_buffer, a_offset, a_ld, complex_one, c_buffer, c_offset, c_ld,
         event_, true);
}

// =================================================================================================

// Compiles the templated class
template class Xher2k<float2,float>;
template class Xher2k<double2,double>;

// =================================================================================================
}} // namespace gpgpu::blas
