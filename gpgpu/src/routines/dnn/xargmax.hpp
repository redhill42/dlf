#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xargmax : public blas::Routine {
public:
    Xargmax(const Queue& queue, Event* event, const std::string& name = "ARGMAX");

    void DoArgMax(const size_t m, const size_t k, const size_t n,
                  const Buffer<T>& x_buffer, Buffer<int>& y_buffer);

    void DoArgMin(const size_t m, const size_t k, const size_t n,
                  const Buffer<T>& x_buffer, Buffer<int>& y_buffer);

};

}} // namespace gpgpu::dnn
