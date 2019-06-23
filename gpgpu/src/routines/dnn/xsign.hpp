#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xsign : public blas::Routine {
public:
    Xsign(const Queue& queue, Event* event, const std::string& name = "SIGN");

    void DoSign(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc);
};

}} // namespace gpgpu::dnn
