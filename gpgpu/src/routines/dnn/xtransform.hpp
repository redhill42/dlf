#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xtransform : public blas::Routine {
public:
    Xtransform(const Queue& queue, Event* event, const std::string& name);

    void DoTransform(const size_t n,
        const Buffer<T>& x_buffer, const size_t x_offset,
        Buffer<T>& y_buffer, const size_t y_offset);

    void DoTransform(const size_t n, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
        Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride);
};

}} // namespace gpgpu::dnn
