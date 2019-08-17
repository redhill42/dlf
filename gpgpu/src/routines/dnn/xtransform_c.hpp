#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {
template <typename T, typename R>
class Xtransform_c : public blas::Routine {
public:
    Xtransform_c(const Queue& queue, Event* event, const std::string& name = "TRANSFORM_C");

    void DoTransform(const std::string& name,
        const size_t m, const size_t n, const size_t channels,
        const Buffer<T>& x_buffer, const Buffer<T>& y_buffer,
        Buffer<R>& z_buffer);
};

}} // namespace gpgpu::dnn
