#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xargreduce : public blas::Routine {
public:
    Xargreduce(const Queue& queue, Event* event, const std::string& name = "ARGREDUCE");

    void DoArgReduce(const std::string& name, const size_t n, const size_t k,
                     const std::vector<size_t>& dims, const std::vector<size_t>& strides,
                     const Buffer<T>& x_buffer, const size_t x_offset,
                     Buffer<int>& y_buffer);
};

}} // namespace gpgpu::dnn
