
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xsyr2k class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xsyr2k.hpp"
#include "routines/level3/xgemm.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xsyr2k<T>::Xsyr2k(const Queue &queue, Event* event, const std::string &name):
    Xsyrk<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xsyr2k<T>::DoSyr2k(const Layout layout, const Triangle triangle, const Transpose ab_transpose,
                        const size_t n, const size_t k,
                        const T alpha,
                        const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                        const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                        const T beta,
                        Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Runs the first matrix multiplication
  const auto negated_ab_transpose = (ab_transpose != Transpose::NoTrans) ? Transpose::NoTrans : Transpose::Trans;
  SyrkAB(layout, triangle, ab_transpose, negated_ab_transpose, n, k, alpha,
         a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld,
         nullptr);

  // Swaps the arguments for matrices A and B, and sets 'beta' to 1
  auto one = ConstantOne<T>();
  SyrkAB(layout, triangle, ab_transpose, negated_ab_transpose, n, k, alpha,
         b_buffer, b_offset, b_ld, a_buffer, a_offset, a_ld, one, c_buffer, c_offset, c_ld,
         event_);
}

// =================================================================================================

// Compiles the templated class
template class Xsyr2k<half>;
template class Xsyr2k<float>;
template class Xsyr2k<double>;
template class Xsyr2k<float2>;
template class Xsyr2k<double2>;

// =================================================================================================
}} // namespace gpgpu::blas
