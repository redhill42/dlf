
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xspmv class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level2/xspmv.hpp"

#include <string>
#include <vector>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xspmv<T>::Xspmv(const Queue &queue, Event* event, const std::string &name):
    Xgemv<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xspmv<T>::DoSpmv(const Layout layout, const Triangle triangle,
                      const size_t n,
                      const T alpha,
                      const Buffer<T> &ap_buffer, const size_t ap_offset,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const T beta,
                      Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc) {

  // The data is either in the upper or lower triangle
  size_t is_upper = ((triangle == Triangle::Upper && layout != Layout::RowMajor) ||
                     (triangle == Triangle::Lower && layout == Layout::RowMajor));

  // Runs the generic matrix-vector multiplication, disabling the use of fast vectorized kernels.
  // The specific symmetric packed matrix-accesses are implemented in the kernel guarded by the
  // ROUTINE_SPMV define.
  bool fast_kernels = false;
  MatVec(layout, Transpose::NoTrans,
         n, n, alpha,
         ap_buffer, ap_offset, n,
         x_buffer, x_offset, x_inc, beta,
         y_buffer, y_offset, y_inc,
         fast_kernels, fast_kernels,
         is_upper, true, 0, 0);
}

// =================================================================================================

// Compiles the templated class
template class Xspmv<half>;
template class Xspmv<float>;
template class Xspmv<double>;

// =================================================================================================
}} // namespace gpgpu::blas
