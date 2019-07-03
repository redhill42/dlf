#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xtransform_b : public blas::Routine {
public:
    Xtransform_b(const Queue& queue, Event* event, const std::string& name = "TRANSFORM_B");

    void DoTransform(const std::string& name,
        const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
        const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
        Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc);

    void DoTransform(const std::string& name, const size_t n,
        const Buffer<T>& x_buffer, const Buffer<T>& y_buffer, Buffer<T>& z_buffer,
        const std::vector<size_t>& lstride, const std::vector<size_t>& rstride,
        const std::vector<size_t>& oshape);
};

}}
