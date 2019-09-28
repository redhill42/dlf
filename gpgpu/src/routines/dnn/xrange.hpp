#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xrange : public blas::Routine {
public:
    Xrange(const Queue& queue, Event* event, const std::string& name = "RANGE");

    void DoRange(const size_t n, const T start, const T delta,
                 Buffer<T>& x_buffer, const size_t x_offset);

    void DoRangeStrided(
                 const size_t n, const T start, const T delta,
                 const std::vector<size_t>& dims, const std::vector<size_t>& strides,
                 Buffer<T>& x_buffer, const size_t x_offset);
};

}} // namespace gpgpu::dnn
