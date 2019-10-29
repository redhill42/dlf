#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T, typename R>
class Xtransform_b : public blas::Routine {
public:
    Xtransform_b(const Queue& queue, Event* event, const std::string& name);

    void DoTransform(
        const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset,
        const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset,
        Buffer<R>& z_buffer, const size_t z_offset);

    void DoTransformStrided(
        const size_t n, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
        const Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride,
        Buffer<R>& z_buffer, const size_t z_offset, const std::vector<size_t>& z_stride);

    void DoTransformChannel(
        const size_t m, const size_t n, const size_t channels,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const Buffer<T>& y_buffer, const size_t y_offset,
        Buffer<R>& z_buffer, const size_t z_offset);
};

}} // namespace gpgpu::dnn
