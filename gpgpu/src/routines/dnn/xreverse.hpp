#pragma once
#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xreverse : public blas::Routine {
public:
    Xreverse(const Queue& queue, Event* event, const std::string& name = "REVERSE");

    void DoReverse(const size_t m, const size_t n, 
                   const std::vector<size_t>& dims, const std::vector<size_t>& strides,
                   Buffer<T>& x_buffer, const size_t x_offset);
};

}} // namespace gpgpu::dnn
