#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xtransform_p : public blas::Routine {
public:
    Xtransform_p(const Queue& queue, Event* event, const std::string& name = "TRANSFORM_P");

    void DoTransform(const std::string& name, const size_t n, const T alpha, const T beta,
                     const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                     Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc);
};

}} // namespace gpgpu::dnn
