#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xneg : public blas::Routine {
public:
    Xneg(const Queue& queue, Event* event, const std::string& name = "NEG");

    void DoNeg(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
               Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc);
};

}} // namespace gpgpu::dnn
