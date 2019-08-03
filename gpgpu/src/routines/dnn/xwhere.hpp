#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xwhere : public blas::Routine {
public:
    Xwhere(const Queue& queue, Event* event, const std::string& name = "WHERE");

    void DoWhere(
        const size_t n, const size_t rank,
        const Buffer<bool>& c_buffer, const size_t c_offset,
        const std::vector<size_t>& c_dim, const std::vector<size_t>& c_stride,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
        const Buffer<T>& y_buffer, const size_t y_offset,
        const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
        Buffer<T>& z_buffer, const size_t z_offset);
};

}} // namespace gpgpu::dnn