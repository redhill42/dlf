
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xdotu class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level1/xdotu.hpp"

#include <string>

namespace gpgpu { namespace blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xdotu<T>::Xdotu(const Queue &queue, Event* event, const std::string &name):
    Xdot<T>(queue, event, name) {
}

// =================================================================================================

// The main routine
template <typename T>
void Xdotu<T>::DoDotu(const size_t n,
                      const Buffer<T> &x_buffer, const size_t x_offset, const size_t x_inc,
                      const Buffer<T> &y_buffer, const size_t y_offset, const size_t y_inc,
                      Buffer<T> &dot_buffer, const size_t dot_offset) {
  DoDot(n,
        x_buffer, x_offset, x_inc,
        y_buffer, y_offset, y_inc,
        dot_buffer, dot_offset,
        false);
}

// =================================================================================================

// Compiles the templated class
template class Xdotu<float2>;
template class Xdotu<double2>;

// =================================================================================================
}} // namespace gpgpu::blas
