
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the common code to perform (partial) matrix inverting.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_XINVERT_H_
#define GPGPU_BLAS_ROUTINES_XINVERT_H_

#include "routine.hpp"

namespace gpgpu { namespace blas {

template <typename T>
class Xinvert: public Routine {
public:

    // Constructor
    Xinvert(const Queue &queue, Event* event, const std::string &name = "INVERT");

    // Inverts diagonal square blocks of a matrix
    void InvertMatrixDiagonalBlocks(
        const Layout layout, const Triangle triangle, const Diagonal diag,
        const size_t n, const size_t block_size,
        const Buffer<T>& src, const size_t src_offset, const size_t ld_src,
        Buffer<T>& dest, const size_t dest_offset);
};

}} // namespace gpgpu::blas

#endif // GPGPU_BLAS_ROUTINES_XINVERT_H_
