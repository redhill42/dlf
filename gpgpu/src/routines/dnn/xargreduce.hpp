#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xargreduce : public blas::Routine {
public:
    Xargreduce(const Queue& queue, Event* event, const std::string& name);

    void DoArgReduce(const size_t n, const size_t k,
                     const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
                     const Buffer<T>& x_buffer, const size_t x_offset,
                     const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
                     Buffer<int>& y_buffer, const size_t y_offset);
};

}} // namespace gpgpu::dnn
