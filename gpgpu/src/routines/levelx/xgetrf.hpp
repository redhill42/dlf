#pragma once

#include "routine.hpp"

namespace gpgpu { namespace blas {

template <typename T>
class Xgetrf : public blas::Routine {
public:
    Xgetrf(const Queue& queue, Event* event, const std::string& name = "GETRF");

    void DoGetrf(const size_t m, const size_t n,
                 Buffer<T>& A, const size_t a_offset, const size_t lda,
                 Buffer<int32_t>& ipiv, const size_t ipiv_offset);

    void DoGetrs(Transpose trans, size_t n, size_t nrhs,
                 const Buffer<T>& A, const size_t a_offset, const size_t lda,
                 const Buffer<int32_t>& ipiv, const size_t ipiv_offset,
                 Buffer<T>& B, const size_t b_offset, const size_t ldb);

    void DoLaswp(size_t n, Buffer<T>& A, const size_t a_offset, const size_t lda,
                 size_t k1, size_t k2,
                 const Buffer<int32_t>& ipiv, const size_t ip_offset, int ip_inc);
};

}} // namespace gpgpu::blas
