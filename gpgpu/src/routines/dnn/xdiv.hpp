#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xdiv : public blas::Routine {
public:
    Xdiv(const Queue& queue, Event* event, const std::string& name = "DIV");

    void DoDiv(const size_t n,
               const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
               const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
               Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc);
};

}} // namespace gpgpu::dnn