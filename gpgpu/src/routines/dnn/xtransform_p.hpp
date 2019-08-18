#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xtransform_p : public blas::Routine {
public:
    Xtransform_p(const Queue& queue, Event* event, const std::string& name = "TRANSFORM_P");

    void DoTransform(const std::string& name, const T alpha, const T beta, const size_t n,
                     const Buffer<T>& x_buffer, const size_t x_offset,
                     Buffer<T>& y_buffer, const size_t y_offset);

    void DoTransform(const std::string& name, const T alpha, const T beta,
        const size_t n, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
        Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride);
};

}} // namespace gpgpu::dnn
