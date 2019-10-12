#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xresize : public blas::Routine {
public:
    Xresize(const Queue& queue, Event* event, const std::string& name = "RESIZE");

    void DoResize1D(const size_t batch_count,
                    const Buffer<T>& x_buffer, const size_t x_offset,
                    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
                    Buffer<T>& y_buffer, const size_t y_offset,
                    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides);

    void DoResize2D(const size_t batch_count,
                    const Buffer<T>& x_buffer, const size_t x_offset,
                    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
                    Buffer<T>& y_buffer, const size_t y_offset,
                    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides);
};

}} // namespace gpgpu::dnn
