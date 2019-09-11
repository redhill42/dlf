#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xreduce : public blas::Routine {
public:
    Xreduce(const Queue& queue, Event* event, const std::string& name = "REDUCE");

    void DoReduce(const std::string& name, const size_t m, const size_t n,
                  const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
                  const Buffer<T>& x_buffer, const size_t x_offset,
                  const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
                  Buffer<T>& y_buffer, const size_t y_offset);
};

}} // namespace gpgpu::dnn
